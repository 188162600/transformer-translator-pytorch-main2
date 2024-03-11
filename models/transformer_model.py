import copy
from typing import Optional, Any, Union, Callable

import torch
import warnings
from torch import Tensor
from torch import nn


from .transformer_task import TransformerTask,TransformerTask2,TransformerTask3
from meta.section import Section
from  torch.nn import LayerNorm
from torch.nn.init import xavier_uniform_
from .transformer_task import TransformerTask
from utils.utils import label_smoothing
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
class TransformerModel:

    def __init__(self,opt) -> None:
        self.opt = opt
        self.all_data=dict()

        self.creterian = nn.KLDivLoss(reduction='batchmean')
        self.task=TransformerTask(opt)

    def set_input(self,inputs):
        self.inputs=inputs
    def forward(self):
        b_text_src, b_text_trg, b_mask_src, b_mask_trg = self.inputs

        return self.task.forward(b_text_src, b_text_trg[:,:-1], b_mask_src, b_mask_trg[:,:-1])



    def eval(self):
        with torch.no_grad():
            b_text_src, b_text_trg, b_mask_src, b_mask_trg = self.inputs

            b_predicted_log_distributions,steps=self.forward()
            b_smooth_label = label_smoothing(b_text_trg[:,1:], self.opt.trg_vocab_size, self.opt.num_special_tokens_trg)
            loss = self.creterian(b_predicted_log_distributions, b_smooth_label)


            # loss.backward()
            # optimizer.step()

            # loss_sum_train += loss_scalar
            # step_losses.append(loss_scalar)

            loss_scalar = loss.item()
            b_predictions = torch.argmax(b_predicted_log_distributions, dim=2)

            err = (b_predictions != b_text_trg[:,1:]).sum().item()
            num_tokens = torch.sum(~b_mask_trg[:,:-1]).item()
            return loss_scalar, err, num_tokens

    def optimize_parameters(self):
        b_text_src, b_text_trg, b_mask_src, b_mask_trg = self.inputs

        b_predicted_log_distributions,steps=self.forward()
        b_smooth_label = label_smoothing(b_text_trg[:,1:], self.opt.trg_vocab_size, self.opt.num_special_tokens_trg)
        loss = self.creterian(b_predicted_log_distributions, b_smooth_label)

        self.task.optimize_layers(loss)
        self.task.optimize_classifier(loss,steps)
        # loss.backward()
        # optimizer.step()

        # loss_sum_train += loss_scalar
        # step_losses.append(loss_scalar)

        loss_scalar = loss.item()
        b_predictions = torch.argmax(b_predicted_log_distributions, dim=2)

        err = (b_predictions != b_text_trg[:,1:]).sum().item()
        num_tokens = torch.sum(~b_mask_trg[:,:-1]).item()
        return loss_scalar, err, num_tokens
    def define_optimizer(self,param_optimizer,num_training_steps):
        no_decay = ['bias', 'LayerNorm.weight']
                # Separate the `weight` parameters from the `bias` parameters.
                # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
                # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
        optimizer_grouped_parameters = [
            # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.1},

            # Filter for parameters which *do* include those.
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
        # Note - `optimizer_grouped_parameters` only includes the parameter values, not
        # the names.

        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr = self.opt.learning_rate,
                eps = 1e-8
                )

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = 0, # Default value in run_glue.py
                num_training_steps = num_training_steps
                )
        return optimizer,scheduler


    def setup(self):
        self.task.setup()
        # if self.opt.is_train and self.opt.continue_train:
        #     self.load_network()
        # else:
        #     self.task.setup()
        #
        #     for index,section in enumerate( self.task.sections):
        #         param_optimizer = list(section.named_parameters())
        #         optimizer,scheduler=self.define_optimizer(param_optimizer, self.opt.dataloader_len * self.opt.num_of_epochs)
        #         self.task.set_optimizer(index,optimizer,scheduler)
        #         param_optimizer=list(self.task.steps_classifiers[index].named_parameters())
        #         optimizer,scheduler=self.define_optimizer(param_optimizer, self.opt.dataloader_len * self.opt.num_of_epochs)
        #         self.task.set_classifier_optimizer(index,optimizer,scheduler)
    def load_network(self):

        self.task.load_network(set())
    def save_network(self):
        self.task.save_network(set())






class TransformerModel2:

    def __init__(self,opt) -> None:
        self.opt = opt
        self.all_data=dict()

        self.creterian = nn.KLDivLoss(reduction='batchmean')
        self.task=TransformerTask3(opt)

    def set_input(self,inputs):
        self.inputs=inputs
    def forward(self):
        b_text_src, b_text_trg, b_mask_src, b_mask_trg = self.inputs

        return self.task.forward(b_text_src, b_text_trg[:,:-1], b_mask_src, b_mask_trg[:,:-1])
#             batch = tuple(t.to(device) for t in batch)
#             b_text_src, b_text_trg, b_mask_src, b_mask_trg = batch

#             optimizer.zero_grad()
#             b_predicted_log_distributions = model(b_text_src, b_text_trg[:,:-1], b_mask_src, b_mask_trg[:,:-1])
#             b_smooth_label = label_smoothing(b_text_trg[:,1:], training_config["trg_vocab_size"], training_config["num_special_tokens_trg"])



    def eval(self):
        with torch.no_grad():
            b_text_src, b_text_trg, b_mask_src, b_mask_trg = self.inputs

            b_predicted_log_distributions,steps=self.forward()
            b_smooth_label = label_smoothing(b_text_trg[:,1:], self.opt.trg_vocab_size, self.opt.num_special_tokens_trg)
            loss = self.creterian(b_predicted_log_distributions, b_smooth_label)


            # loss.backward()
            # optimizer.step()

            # loss_sum_train += loss_scalar
            # step_losses.append(loss_scalar)

            loss_scalar = loss.item()
            b_predictions = torch.argmax(b_predicted_log_distributions, dim=2)

            err = (b_predictions != b_text_trg[:,1:]).sum().item()
            num_tokens = torch.sum(~b_mask_trg[:,:-1]).item()
            return loss_scalar, err, num_tokens

    def optimize_parameters(self):
        b_text_src, b_text_trg, b_mask_src, b_mask_trg = self.inputs

        self.task.layers_optimizer.zero_grad()
        b_predicted_log_distributions,steps=self.forward()
        
        #             batch = tuple(t.to(device) for t in batch)
#             b_text_src, b_text_trg, b_mask_src, b_mask_trg = batch

#             optimizer.zero_grad()
#             b_predicted_log_distributions = model(b_text_src, b_text_trg[:,:-1], b_mask_src, b_mask_trg[:,:-1])
#             b_smooth_label = label_smoothing(b_text_trg[:,1:], training_config["trg_vocab_size"], training_config["num_special_tokens_trg"])
        
#             loss = creterian(b_predicted_log_distributions, b_smooth_label)

#             loss.backward()
#             optimizer.step()


        b_smooth_label = label_smoothing(b_text_trg[:,1:], self.opt.trg_vocab_size, self.opt.num_special_tokens_trg)
        loss = self.creterian(b_predicted_log_distributions, b_smooth_label)
        loss.backward()
        self.task.layers_optimizer.step()

        # self.task.optimize_layers(loss)
        # self.task.optimize_classifier(loss,steps)
        # loss.backward()
        # optimizer.step()

        # loss_sum_train += loss_scalar
        # step_losses.append(loss_scalar)

        loss_scalar = loss.item()
        b_predictions = torch.argmax(b_predicted_log_distributions, dim=2)

        err = (b_predictions != b_text_trg[:,1:]).sum().item()
        num_tokens = torch.sum(~b_mask_trg[:,:-1]).item()
        return loss_scalar, err, num_tokens
    def define_optimizer(self,param_optimizer,num_training_steps):
        no_decay = ['bias', 'LayerNorm.weight']
                # Separate the `weight` parameters from the `bias` parameters.
                # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
                # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
        optimizer_grouped_parameters = [
            # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.1},

            # Filter for parameters which *do* include those.
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
        # Note - `optimizer_grouped_parameters` only includes the parameter values, not
        # the names.

        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr = self.opt.learning_rate,
                eps = 1e-8
                )

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = 0, # Default value in run_glue.py
                num_training_steps = num_training_steps
                )
        return optimizer,scheduler


    def setup(self):
        self.task.setup()
        # if self.opt.is_train and self.opt.continue_train:
        #     self.load_network()
        # else:
        #     self.task.setup()
        #
        #     for index,section in enumerate( self.task.sections):
        #         param_optimizer = list(section.named_parameters())
        #         optimizer,scheduler=self.define_optimizer(param_optimizer, self.opt.dataloader_len * self.opt.num_of_epochs)
        #         self.task.set_optimizer(index,optimizer,scheduler)
        #         param_optimizer=list(self.task.steps_classifiers[index].named_parameters())
        #         optimizer,scheduler=self.define_optimizer(param_optimizer, self.opt.dataloader_len * self.opt.num_of_epochs)
        #         self.task.set_classifier_optimizer(index,optimizer,scheduler)
    def load_network(self):

        self.task.load_network(set())
    def save_network(self):
        self.task.save_network(set())









class TransformerModel3:

    def __init__(self,opt) -> None:
        self.opt = opt
        self.all_data=dict()

        self.creterian = nn.KLDivLoss(reduction='batchmean')
        self.task=TransformerTask2(opt)

    def set_input(self,inputs):
        self.inputs=inputs
    def forward(self):
        b_text_src, b_text_trg, b_mask_src, b_mask_trg = self.inputs

        return self.task.forward(b_text_src, b_text_trg[:,:-1], b_mask_src, b_mask_trg[:,:-1])
#             batch = tuple(t.to(device) for t in batch)
#             b_text_src, b_text_trg, b_mask_src, b_mask_trg = batch

#             optimizer.zero_grad()
#             b_predicted_log_distributions = model(b_text_src, b_text_trg[:,:-1], b_mask_src, b_mask_trg[:,:-1])
#             b_smooth_label = label_smoothing(b_text_trg[:,1:], training_config["trg_vocab_size"], training_config["num_special_tokens_trg"])



    def eval(self):
        with torch.no_grad():
            b_text_src, b_text_trg, b_mask_src, b_mask_trg = self.inputs

            b_predicted_log_distributions,steps=self.forward()
            b_smooth_label = label_smoothing(b_text_trg[:,1:], self.opt.trg_vocab_size, self.opt.num_special_tokens_trg)
            loss = self.creterian(b_predicted_log_distributions, b_smooth_label)


            # loss.backward()
            # optimizer.step()

            # loss_sum_train += loss_scalar
            # step_losses.append(loss_scalar)

            loss_scalar = loss.item()
            b_predictions = torch.argmax(b_predicted_log_distributions, dim=2)

            err = (b_predictions != b_text_trg[:,1:]).sum().item()
            num_tokens = torch.sum(~b_mask_trg[:,:-1]).item()
            return loss_scalar, err, num_tokens

    def optimize_parameters(self):
        b_text_src, b_text_trg, b_mask_src, b_mask_trg = self.inputs

        self.task.layers_optimizer.zero_grad()
        b_predicted_log_distributions,steps=self.forward()
        
        #             batch = tuple(t.to(device) for t in batch)
#             b_text_src, b_text_trg, b_mask_src, b_mask_trg = batch

#             optimizer.zero_grad()
#             b_predicted_log_distributions = model(b_text_src, b_text_trg[:,:-1], b_mask_src, b_mask_trg[:,:-1])
#             b_smooth_label = label_smoothing(b_text_trg[:,1:], training_config["trg_vocab_size"], training_config["num_special_tokens_trg"])
        
#             loss = creterian(b_predicted_log_distributions, b_smooth_label)

#             loss.backward()
#             optimizer.step()


        b_smooth_label = label_smoothing(b_text_trg[:,1:], self.opt.trg_vocab_size, self.opt.num_special_tokens_trg)
        loss = self.creterian(b_predicted_log_distributions, b_smooth_label)
        loss.backward()
        self.task.layers_optimizer.step()

        # self.task.optimize_layers(loss)
        # self.task.optimize_classifier(loss,steps)
        # loss.backward()
        # optimizer.step()

        # loss_sum_train += loss_scalar
        # step_losses.append(loss_scalar)

        loss_scalar = loss.item()
        b_predictions = torch.argmax(b_predicted_log_distributions, dim=2)

        err = (b_predictions != b_text_trg[:,1:]).sum().item()
        num_tokens = torch.sum(~b_mask_trg[:,:-1]).item()
        return loss_scalar, err, num_tokens
    def define_optimizer(self,param_optimizer,num_training_steps):
        no_decay = ['bias', 'LayerNorm.weight']
                # Separate the `weight` parameters from the `bias` parameters.
                # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
                # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
        optimizer_grouped_parameters = [
            # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.1},

            # Filter for parameters which *do* include those.
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
        # Note - `optimizer_grouped_parameters` only includes the parameter values, not
        # the names.

        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr = self.opt.learning_rate,
                eps = 1e-8
                )

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps = 0, # Default value in run_glue.py
                num_training_steps = num_training_steps
                )
        return optimizer,scheduler


    def setup(self):
        self.task.setup()
        # if self.opt.is_train and self.opt.continue_train:
        #     self.load_network()
        # else:
        #     self.task.setup()
        #
        #     for index,section in enumerate( self.task.sections):
        #         param_optimizer = list(section.named_parameters())
        #         optimizer,scheduler=self.define_optimizer(param_optimizer, self.opt.dataloader_len * self.opt.num_of_epochs)
        #         self.task.set_optimizer(index,optimizer,scheduler)
        #         param_optimizer=list(self.task.steps_classifiers[index].named_parameters())
        #         optimizer,scheduler=self.define_optimizer(param_optimizer, self.opt.dataloader_len * self.opt.num_of_epochs)
        #         self.task.set_classifier_optimizer(index,optimizer,scheduler)
    def load_network(self):

        self.task.load_network(set())
    def save_network(self):
        self.task.save_network(set())






