import copy
from typing import Optional, Any, Union, Callable

import torch
import warnings
from torch import Tensor
# from .. import functional as F
# from .module import Module
# from .activation import MultiheadAttention
# from .container import ModuleList
# from ..init import xavier_uniform_
# from .dropout import Dropout
# from .linear import Linear

from torch.nn import LayerNorm

from meta.section import SectionLayers
from utils.utils import get_seq_len, detect_is_causal_mask
class TransformerDecoderLayers(SectionLayers):
    r"""TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    __constants__ = ['norm']

    def __init__(self, num_options_each_layer,num_shared_layers,decoder_layer, num_layers, norm=None):
        super().__init__("decoder",num_options_each_layer,[],lambda x:False)
        self.append_layer(decoder_layer)
        for i in range(num_layers-1):
            self.append_shared_layers(0,num_shared_layers)
        
        #self.num_layers = num_layers
        self.norm = norm

    def forward(self, next_steps, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) :
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        seq_len = get_seq_len(tgt, self.base_layers[0].self_attn.batch_first)
        tgt_is_causal = detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)
        def update(__output,__args,__kwargs):

            return (__output,*__args[1:]),__kwargs

        output, last_features=super().forward_with_update(next_steps,update,output,memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal)
        # for mod in self.layers:
        #     output = mod(output, memory, tgt_mask=tgt_mask,
        #                  memory_mask=memory_mask,
        #                  tgt_key_padding_mask=tgt_key_padding_mask,
        #                  memory_key_padding_mask=memory_key_padding_mask,
        #                  tgt_is_causal=tgt_is_causal,
        #                  memory_is_causal=memory_is_causal)
        output=output[0]
        if self.norm is not None:
            output = self.norm(output)

        return  output,last_features
