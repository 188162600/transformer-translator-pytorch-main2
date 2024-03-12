import copy
from typing import Optional, Any, Union, Callable

import torch
import warnings
from torch import Tensor
from torch import nn

from .transformer_decoder import TransformerDecoderLayers
from .transformer_encoder import TransformerEncoderLayers
from .output import OutputSection
from meta.section import Section
from torch.nn import LayerNorm
from torch.nn.init import xavier_uniform_
from utils import utils
from models.embeding import Embedding, PositionalEncoding
from torch.nn import functional as F
from meta.next_steps import LinearStepClassifier,LSTMNextStepClassifier
from utils.utils import get_causal_mask
from itertools import chain
from utils.utils import get_linear_schedule_with_warmup
from torch.optim import AdamW
# from models2.model import Transformer as Transformer2

class TransformerTask:

    def __init__(self, opt, encoder=None, decoder=None) -> None:

        self.opt = opt
        self.src_embedding = Embedding(opt.src_vocab_size, opt.model_dimension)
        self.trg_embedding = Embedding(opt.trg_vocab_size, opt.model_dimension)

        #  positional encoding
        self.src_pos_encoding = PositionalEncoding(model_dimension=opt.model_dimension,
                                                   dropout_probability=opt.dropout_probability, expected_max_sequence_length=opt.max_src_seq_length)
        self.trg_pos_encoding = PositionalEncoding(model_dimension=opt.model_dimension,
                                                   dropout_probability=opt.dropout_probability, expected_max_sequence_length=opt.max_src_seq_length)
        self.hidden_long_term=dict()

        def detach(args, kwargs):
            detached_args = []
            detached_kwargs = dict()
            for arg in args:
                detached_args.append(arg.detach() if isinstance(arg, torch.Tensor) else arg)
            for key in kwargs:
                arg = kwargs[key]
                detached_kwargs[key] = arg.detach() if isinstance(arg, torch.Tensor) else arg

            return detached_args, detached_kwargs

        if encoder is not None:
            self.encoder = encoder
        else:

            encoder_layer = nn.TransformerEncoderLayer(opt.model_dimension, opt.number_of_heads, opt.dim_feedforward,
                                                       opt.dropout_probability, batch_first=True)

            encoder_norm = LayerNorm(opt.model_dimension)
            layers = TransformerEncoderLayers(opt.num_encoder_options_each_layer, opt.num_encoder_shared_layers,
                                              encoder_layer, opt.num_encoder_layers, encoder_norm)
            step_classifier_encoder = nn.TransformerEncoder(encoder_layer, opt.num_classifier_encoder_layers,
                                                            encoder_norm)
            self._reset_parameters(step_classifier_encoder.parameters())

            step_classifier = LSTMNextStepClassifier(opt.num_encoder_layers, opt.num_encoder_options_each_layer,
                                                   (opt.model_dimension,), device=self.opt.device)
            self.encoder = Section(layers, step_classifier_encoder, step_classifier,
                                   lambda src, *args, **kwargs: src[:, 1, :],
                                   detach)
            self._reset_encoder()
        if decoder is not None:
            self.decoder = decoder
        else:
            decoder_layer = nn.TransformerDecoderLayer(opt.model_dimension, opt.number_of_heads, opt.dim_feedforward,
                                                       opt.dropout_probability, batch_first=True)
            decoder_norm = LayerNorm(opt.model_dimension).to(opt.device)
            layers = TransformerDecoderLayers(opt.num_decoder_options_each_layer, opt.num_decoder_shared_layers,
                                              decoder_layer, opt.num_decoder_layers, decoder_norm)
            step_classifier_encoder = None

            step_classifier = LinearStepClassifier(opt.num_decoder_layers, opt.num_decoder_options_each_layer,
                                                   (opt.model_dimension,), device=self.opt.device)

            self.decoder = Section(layers, step_classifier_encoder, step_classifier,
                                   lambda *args, memory, **kwargs: memory[:, 1, :], detach)
            self._reset_decoder()
        # self._reset_parameters()

        self.d_model = opt.model_dimension
        self.nhead = opt.number_of_heads

        self.batch_first = True
        self.linear = nn.Linear(opt.model_dimension, opt.trg_vocab_size).to(opt.device)
        self.log_softmax = nn.LogSoftmax(dim=-1).to(opt.device)
        self.layers = [self.src_embedding, self.trg_embedding, self.src_pos_encoding, self.trg_pos_encoding,
                       self.encoder.layers, self.decoder.layers, self.linear, self.log_softmax]
        self.classifiers = [self.encoder.encoder, self.encoder.classifier, self.decoder.encoder,
                            self.decoder.classifier]
        self.classifiers = list(filter(lambda x: x is not None, self.classifiers))

    def forward(self, b_text_src, b_text_trg, b_mask_src, b_mask_trg):
        # src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
        #         memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
        #         tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
        #         src_is_causal: Optional[bool] = None, tgt_is_causal: Optional[bool] = None,
        #         memory_is_causal: bool = False
        src = self.src_embedding(b_text_src)
        src = self.src_pos_encoding(src)

        trg = self.trg_embedding(b_text_trg)
        trg = self.trg_pos_encoding(trg)

        mask = get_causal_mask(b_text_trg.shape[-1], self.opt.device)
        # transformer_out = self.transformer_body(
        #         src = src,
        #         tgt = trg,
        #         tgt_mask = mask,
        #         src_key_padding_mask = b_mask_src,
        #         tgt_key_padding_mask = b_mask_trg,
        #         memory_key_padding_mask = b_mask_src,
        #         )

        # return trg_log_probs
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != trg.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != trg.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        # transformer_out = self.transformer_body(
        #     src=src_embedding_batch,
        #     tgt=trg_embedding_batch,
        #     tgt_mask=mask,
        #     src_key_padding_mask=b_mask_src,
        #     tgt_key_padding_mask=b_mask_trg,
        #     memory_key_padding_mask=b_mask_src,
        # )
        if src.size(-1) != self.d_model or trg.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        memory, last_feature, encoder_steps = self.encoder.forward(None, self, src, src_key_padding_mask=b_mask_src)
        output, last_feature, decoder_steps = self.decoder.forward(encoder_steps, self, trg, tgt_mask=mask,
                                                                   memory=memory, tgt_key_padding_mask=b_mask_trg,
                                                                   memory_key_padding_mask=b_mask_src)
        
        linear_out = self.linear(output)
        trg_log_probs = self.log_softmax(linear_out)
        return trg_log_probs, (encoder_steps, decoder_steps)

        # return super().custom_forward(args=(src,),forward=(forward_encoder,forward_decoder))

    @staticmethod
    def generate_square_subsequent_mask(
            sz: int,
            device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
            dtype: torch.dtype = torch.get_default_dtype(),
    ) -> Tensor:
        r"""Generate a square causal mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        return utils.generate_square_subsequent_mask(sz, dtype=dtype, device=device)

    def _reset_encoder(self):
        self._reset_parameters(self.encoder.layers.parameters())

    def _reset_decoder(self):
        self._reset_parameters(self.decoder.layers.parameters())

    def _reset_parameters(self, params):
        r"""Initiate parameters in the transformer model."""
        for p in params:
            if p.dim() > 1:
                xavier_uniform_(p)

    def setup(self):
        if self.opt.is_train and self.opt.continue_train:
            self.load_network(self.opt.which_epoch)
        else:
            self.encoder.setup()
            self.decoder.setup()
        for layer in self.layers:
            layer.to(self.opt.device)
        for classifier in self.classifiers:
            classifier.to(self.opt.device)
        # self.trg_embedding.to(self.opt.device)
        # self.src_embedding.to(self.opt.device)
        # self.trg_pos_encoding.to(self.opt.device)
        # self.src_pos_encoding.to(self.opt.device)
        # self.encoder.layers.to(self.opt.device)
        # self.encoder.encoder.to(self.opt.device)
        # self.encoder.classifier.to(self.opt.device)
        # self.decoder.layers.to(self.opt.device)
        # self.decoder.encoder.to(self.opt.device)
        # self.decoder.classifier.to(self.opt.device)
        # self.linear.to(self.opt.device)
        self.layers_optimizer, self.layers_schedular = self.define_optimizer(
            list(chain(*[layer.named_parameters() for layer in self.layers])))

        self.classifier_optimizer, self.classifier_schedular = self.define_optimizer(
            list(chain(*[classifier.named_parameters() for classifier in self.classifiers])))

    def define_optimizer(self, params):

        # param_optimizer = list( chain(self.src_embedding.named_parameters(),self.src_embedding.named_parameters(),self.src_pos_encoding.named_parameters(),self.src_pos_encoding.named_parameters(),self.encoder,self.decoder,self.linear,self. named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        # Separate the `weight` parameters from the `bias` parameters.
        # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
        # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
        optimizer_grouped_parameters = [
            # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.1},

            # Filter for parameters which *do* include those.
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        # Note - `optimizer_grouped_parameters` only includes the parameter values, not
        # the names.

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.opt.learning_rate,
            eps=1e-8
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=self.opt.dataloader_length * self.opt.num_of_epochs
        )
        return optimizer, scheduler

    def optimize_layers(self, loss):
        self.layers_optimizer.zero_grad()
        loss.backward()
        self.layers_optimizer.step()
        self.layers_schedular.step()

    def optimize_classifier(self, loss, steps):
        # (steps[-1]* step.confidence).backward()
        self.classifier_optimizer.zero_grad()

        for step in steps:
            (loss.detach() * step.confidence).backward()
        self.classifier_optimizer.zero_grad()
        self.classifier_optimizer.step()

    def load_network(self, loaded: set):
        self.encoder.load_network(loaded, self.opt.model_load_path, vars(self.opt))
        self.decoder.load_network(loaded, self.opt.model_load_path, vars(self.opt))

    def save_network(self, saved: set):
        self.encoder.save_network(saved, self.opt.model_save_path, vars(self.opt))
        self.decoder.save_network(saved, self.opt.model_save_path, vars(self.opt))


class TransformerTask2:

    def __init__(self, opt, encoder=None, decoder=None) -> None:

        self.opt = opt
        self.src_embedding = Embedding(opt.src_vocab_size, opt.model_dimension)
        self.trg_embedding = Embedding(opt.trg_vocab_size, opt.model_dimension)

        #  positional encoding
        self.src_pos_encoding = PositionalEncoding(model_dimension=opt.model_dimension,
                                                   dropout_probability=opt.dropout_probability)
        self.trg_pos_encoding = PositionalEncoding(model_dimension=opt.model_dimension,
                                                   dropout_probability=opt.dropout_probability)
        
        #self.hidden_long_term=dict()

        def detach(args, kwargs):
            detached_args = []
            detached_kwargs = dict()
            for arg in args:
                detached_args.append(arg.detach() if isinstance(arg, torch.Tensor) else arg)
            for key in kwargs:
                arg = kwargs[key]
                detached_kwargs[key] = arg.detach() if isinstance(arg, torch.Tensor) else arg

            return detached_args, detached_kwargs

        # if encoder is not None:
        #     self.encoder = encoder
        # else:

        #     encoder_layer = nn.TransformerEncoderLayer(opt.model_dimension, opt.number_of_heads, opt.dim_feedforward,
        #                                                opt.dropout_probability, batch_first=True)

        #     encoder_norm = LayerNorm(opt.model_dimension)
        #     self.encoder=nn.TransformerEncoder(encoder_layer, opt.num_encoder_layers, encoder_norm)
        #     self._reset_encoder()
        # if decoder is not None:
        #     self.decoder = decoder
        # else:
        #     decoder_layer = nn.TransformerDecoderLayer(opt.model_dimension, opt.number_of_heads, opt.dim_feedforward,
        #                                                opt.dropout_probability, batch_first=True)
        #     decoder_norm = LayerNorm(opt.model_dimension).to(opt.device)
        #     self.decoder=nn.TransformerDecoder(decoder_layer, opt.num_decoder_layers, decoder_norm)
        #     self._reset_decoder()
        # self._reset_parameters()
        self.transformer_body = nn.Transformer(
                d_model            =opt. model_dimension,
                nhead              = opt.number_of_heads,
                num_encoder_layers = opt.num_encoder_layers,
                num_decoder_layers = opt.num_decoder_layers,
                dropout            = opt.dropout_probability,
                batch_first= True
                )

        self.d_model = opt.model_dimension
        self.nhead = opt.number_of_heads

        self.batch_first = True
        self.linear = nn.Linear(opt.model_dimension, opt.trg_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        
        self.layers = [self.src_embedding, self.trg_embedding, self.src_pos_encoding, self.trg_pos_encoding,
                       self.transformer_body, self.linear, self.log_softmax]
        # self.classifiers = [self.encoder.encoder, self.encoder.classifier, self.decoder.encoder,
        #                     self.decoder.classifier]
        # self.classifiers = list(filter(lambda x: x is not None, self.classifiers))

    def forward(self, b_text_src, b_text_trg, b_mask_src, b_mask_trg):
        # src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
        #         memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
        #         tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
        #         src_is_causal: Optional[bool] = None, tgt_is_causal: Optional[bool] = None,
        #         memory_is_causal: bool = False
        src_embedding_batch = self.src_embedding(b_text_src)
        src_embedding_batch = self.src_pos_encoding(src_embedding_batch)

        trg_embedding_batch = self.trg_embedding(b_text_trg)
        trg_embedding_batch = self.trg_pos_encoding(trg_embedding_batch)


        mask = get_causal_mask(b_text_trg.shape[-1], self.opt.device)
        #mask = get_causal_mask(b_text_trg.shape[-1])
        transformer_out = self.transformer_body(
                src = src_embedding_batch,
                tgt = trg_embedding_batch,
                tgt_mask = mask,
                src_key_padding_mask = b_mask_src,
                tgt_key_padding_mask = b_mask_trg,
                memory_key_padding_mask = b_mask_src,
                )

        linear_out = self.linear(transformer_out)
        trg_log_probs = self.log_softmax(linear_out)
        return trg_log_probs,()
        # transformer_out = self.transformer_body(
        #         src = src,
        #         tgt = trg,
        #         tgt_mask = mask,
        #         src_key_padding_mask = b_mask_src,
        #         tgt_key_padding_mask = b_mask_trg,
        #         memory_key_padding_mask = b_mask_src,
        #         )

        # return trg_log_probs
        # is_batched = src.dim() == 3
        # if not self.batch_first and src.size(1) != trg.size(1) and is_batched:
        #     raise RuntimeError("the batch number of src and tgt must be equal")
        # elif self.batch_first and src.size(0) != trg.size(0) and is_batched:
        #     raise RuntimeError("the batch number of src and tgt must be equal")
        # # transformer_out = self.transformer_body(
        # #     src=src_embedding_batch,
        # #     tgt=trg_embedding_batch,
        # #     tgt_mask=mask,
        # #     src_key_padding_mask=b_mask_src,
        # #     tgt_key_padding_mask=b_mask_trg,
        # #     memory_key_padding_mask=b_mask_src,
        # # )
        # if src.size(-1) != self.d_model or trg.size(-1) != self.d_model:
        #     raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        output = self.transformer_body(
                src = src,
                tgt = trg,
                tgt_mask = mask,
                src_key_padding_mask = b_mask_src,
                tgt_key_padding_mask = b_mask_trg,
                memory_key_padding_mask = b_mask_src,
                )

        # memory=self.encoder.forward(src,src_key_padding_mask=b_mask_src)
        # output=self.decoder.forward(trg,memory,tgt_mask=mask,tgt_key_padding_mask=b_mask_trg,memory_key_padding_mask=b_mask_src)
        # memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
        #                       is_causal=src_is_causal)
        # output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
        #                       tgt_key_padding_mask=tgt_key_padding_mask,
        #                       memory_key_padding_mask=memory_key_padding_mask,
        #                       tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
        
        linear_out = self.linear(output)
        trg_log_probs = self.log_softmax(linear_out)
        return trg_log_probs, ()

        # return super().custom_forward(args=(src,),forward=(forward_encoder,forward_decoder))

    @staticmethod
    def generate_square_subsequent_mask(
            sz: int,
            device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
            dtype: torch.dtype = torch.get_default_dtype(),
    ) -> Tensor:
        r"""Generate a square causal mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        return utils.generate_square_subsequent_mask(sz, dtype=dtype, device=device)

    def _reset_encoder(self):
        self._reset_parameters(self.encoder.layers.parameters())

    def _reset_decoder(self):
        self._reset_parameters(self.decoder.layers.parameters())

    def _reset_parameters(self, params):
        r"""Initiate parameters in the transformer model."""
        for p in params:
            if p.dim() > 1:
                xavier_uniform_(p)

    def setup(self):
        if self.opt.is_train and self.opt.continue_train:
            self.load_network(self.opt.which_epoch)
        else:
            pass
            # self.encoder.setup()
            # self.decoder.setup()
        for layer in self.layers:
            layer.to(self.opt.device)
        # for classifier in self.classifiers:
        #     classifier.to(self.opt.device)
        
        # self.trg_embedding.to(self.opt.device)
        # self.src_embedding.to(self.opt.device)
        # self.trg_pos_encoding.to(self.opt.device)
        # self.src_pos_encoding.to(self.opt.device)
        # self.encoder.layers.to(self.opt.device)
        # self.encoder.encoder.to(self.opt.device)
        # self.encoder.classifier.to(self.opt.device)
        # self.decoder.layers.to(self.opt.device)
        # self.decoder.encoder.to(self.opt.device)
        # self.decoder.classifier.to(self.opt.device)
        # self.linear.to(self.opt.device)
        self.layers_optimizer, self.layers_schedular = self.define_optimizer(
            list(chain(*[layer.named_parameters() for layer in self.layers])))

        # self.classifier_optimizer, self.classifier_schedular = self.define_optimizer(
        #     list(chain(*[classifier.named_parameters() for classifier in self.classifiers])))

    def define_optimizer(self, params):

        # param_optimizer = list( chain(self.src_embedding.named_parameters(),self.src_embedding.named_parameters(),self.src_pos_encoding.named_parameters(),self.src_pos_encoding.named_parameters(),self.encoder,self.decoder,self.linear,self. named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        # Separate the `weight` parameters from the `bias` parameters.
        # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
        # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
        optimizer_grouped_parameters = [
            # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.1},

            # Filter for parameters which *do* include those.
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        # Note - `optimizer_grouped_parameters` only includes the parameter values, not
        # the names.

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.opt.learning_rate,
            eps=1e-8
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=self.opt.dataloader_length * self.opt.num_of_epochs
        )
        return optimizer, scheduler

    def optimize_layers(self, loss):
        
        self.layers_optimizer.zero_grad()
        loss.backward()
        self.layers_optimizer.step()
        self.layers_schedular.step()

    def optimize_classifier(self, loss, steps):
        return
        # (steps[-1]* step.confidence).backward()
        self.classifier_optimizer.zero_grad()

        for step in steps:
            (loss.detach() * step.confidence).backward()
        self.classifier_optimizer.zero_grad()

    def load_network(self, loaded: set):
        return
        encoder_state_dict = torch.load(self.opt.model_load_path.format_map( vars(self.opt)|{'model_name':'encoder'}))
        self.encoder.load_state_dict(encoder_state_dict)    
        decoder_state_dict = torch.load(self.opt.model_load_path.format_map( vars(self.opt)|{'model_name':'decoder'}))
        self.decoder.load_state_dict(decoder_state_dict)
       
        # self.encoder.load_network(loaded, self.opt.model_load_path, vars(self.opt))
        # self.decoder.load_network(loaded, self.opt.model_load_path, vars(self.opt))

    def save_network(self, saved: set):
        return
        
        
        torch.save(self.encoder.state_dict(),self.opt.model_save_path.format_map( vars(self.opt)|{'model_name':'encoder'}))
        torch.save(self.decoder.state_dict(),self.opt.model_save_path.format_map( vars(self.opt)|{'model_name':'decoder'}))
        # self.encoder.save_network(saved, self.opt.model_save_path, vars(self.opt))
        # self.decoder.save_network(saved, self.opt.model_save_path, vars(self.opt))
class TransformerTask3:
    def __init__(self,opt) -> None:
        self.model=Transformer2(opt.model_dimension, opt.number_of_heads, opt.dim_feedforward, opt.num_encoder_layers, opt.num_decoder_layers, opt.src_vocab_size, opt.trg_vocab_size, opt.dropout_probability, opt.device)
        self.opt=opt
    def save_network(self,loaded:set):
        pass
    def load_network(self,loaded:set):
        pass
    def setup(self):
        no_decay = ['bias', 'LayerNorm.weight']
        # Separate the `weight` parameters from the `bias` parameters.
        # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
        # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
        optimizer_grouped_parameters = [
            # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.1},

            # Filter for parameters which *do* include those.
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        # Note - `optimizer_grouped_parameters` only includes the parameter values, not
        # the names.

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.opt.learning_rate,
            eps=1e-8
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=self.opt.dataloader_length * self.opt.num_of_epochs
        )
    def forward(self, b_text_src, b_text_trg, b_mask_src, b_mask_trg):
            return self.model(b_text_src, b_text_trg, b_mask_src, b_mask_trg)
        
    def optimize_layers(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def save_network(self,loaded:set):
        pass