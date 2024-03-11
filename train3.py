import time
from options.train_options import TrainOptions

from torch import profiler
import torch

from numpy import shape
from tqdm import tqdm
import torch
from torch.cuda import utilization
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
# from src.dataset import parallelCorpus
from utils.utils import pad_to_max_with_mask, label_smoothing
from models.transformer_model import TransformerModel, TransformerModel2, TransformerModel3
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from data.parallel_corpus import ParallelCorpus
import json
import os
import argparse

import os


def load_trails(opt):
    step_losses = list()
    step_losses_path = opt.step_losses_pth.format_map(vars(opt))
    if os.path.exists(step_losses_path):
        with open(step_losses_path, 'r', encoding='UTF-8') as file:
            step_losses = json.load(file)
            file.close()

    train_losses = list()
    train_losses_path = opt.train_losses_pth.format_map(vars(opt))
    if os.path.exists(opt.train_losses_pth):
        with open(train_losses_path, 'r', encoding='UTF-8') as file:
            train_losses = json.load(file)
            file.close()

    eval_losses = list()
    eval_losses_path = opt.eval_losses_pth.format_map(vars(opt))
    if os.path.exists(eval_losses_path):
        with open(eval_losses_path, 'r', encoding='UTF-8') as file:
            eval_losses = json.load(file)
            file.close()

    train_accuracy = list()
    train_accuracy_path = opt.train_accuracy_pth.format_map(vars(opt))
    if os.path.exists(train_accuracy_path):
        with open(train_accuracy_path, 'r', encoding='UTF-8') as file:
            train_accuracy = json.load(file)
            file.close()

    eval_accuracy = list()
    eval_accuracy_path = opt.eval_accuracy_pth.format_map(vars(opt))
    if os.path.exists(eval_accuracy_path):
        with open(eval_accuracy_path, 'r', encoding='UTF-8') as file:
            eval_accuracy = json.load(file)
            file.close()

    return step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy


def save_trails(opt, step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy):
    step_losses_path = opt.step_losses_pth.format_map(vars(opt))
    with open(step_losses_path, 'w', encoding='UTF-8') as file:
        json.dump(step_losses, file)
        file.close()
    train_losses_path = opt.train_losses_pth.format_map(vars(opt))
    with open(train_losses_path, 'w', encoding='UTF-8') as file:
        json.dump(train_losses, file)
        file.close()
    eval_losses_path = opt.eval_losses_pth.format_map(vars(opt))
    with open(eval_losses_path, 'w', encoding='UTF-8') as file:
        json.dump(eval_losses, file)
        file.close()
    train_accuracy_path = opt.train_accuracy_pth.format_map(vars(opt))
    with open(train_accuracy_path, 'w', encoding='UTF-8') as file:
        json.dump(train_accuracy, file)
        file.close()
    eval_accuracy_path = opt.eval_accuracy_pth.format_map(vars(opt))
    with open(eval_accuracy_path, 'w', encoding='UTF-8') as file:
        json.dump(eval_accuracy, file)
        file.close()


def prepare_tokenizer(opt):
    tokenizer_path_src = opt.tokenizer_path_src.format_map(vars(opt))
    tokenizer_path_trg = opt.tokenizer_path_trg.format_map(vars(opt))
    data_path_train_src = opt.data_path_train_src.format_map(vars(opt))
    data_path_train_trg = opt.data_path_train_trg.format_map(vars(opt))

    if not os.path.exists(tokenizer_path_src):

        from tokenizers.implementations import ByteLevelBPETokenizer

        tokenizer_src = ByteLevelBPETokenizer()
        tokenizer_src.train(files=[data_path_train_src], vocab_size=opt.src_vocab_size, min_frequency=2,
                            special_tokens=[
                                "<pad>",
                                "<mask>",
                                "<unk>",
                                "<s>",
                                "</s>",
                            ])
        if not os.path.exists(tokenizer_path_src):
            os.makedirs(tokenizer_path_src)
        tokenizer_src.save_model(tokenizer_path_src)
    if not os.path.exists(tokenizer_path_trg):
        tokenizer_trg = ByteLevelBPETokenizer()
        tokenizer_trg.train(files=[data_path_train_trg], vocab_size=opt.trg_vocab_size, min_frequency=2,
                            special_tokens=[
                                "<pad>",
                                "<mask>",
                                "<unk>",
                                "<s>",
                                "</s>",
                            ])
        if not os.path.exists(tokenizer_path_trg):
            os.makedirs(tokenizer_path_trg)
        tokenizer_trg.save_model(tokenizer_path_trg)
    return tokenizer_path_src, tokenizer_path_trg, data_path_train_src, data_path_train_trg
def prepare_tokenizer(opt):
    tokenizer_path_src = opt.tokenizer_path_src.format_map(vars(opt))
    tokenizer_path_trg = opt.tokenizer_path_trg.format_map(vars(opt))
    data_path_train_src = opt.data_path_train_src.format_map(vars(opt))
    data_path_train_trg = opt.data_path_train_trg.format_map(vars(opt))

    #if not os.path.exists(tokenizer_path_src):

    from tokenizers.implementations import ByteLevelBPETokenizer

    tokenizer_src = ByteLevelBPETokenizer()
    tokenizer_src.train(files=[data_path_train_src], vocab_size=opt.src_vocab_size, min_frequency=2,
                        special_tokens=[
                            "<pad>",
                            "<mask>",
                            "<unk>",
                            "<s>",
                            "</s>",
                        ])
    if not os.path.exists(tokenizer_path_src):
        os.makedirs(tokenizer_path_src)
    tokenizer_src.save_model(tokenizer_path_src)
    #if not os.path.exists(tokenizer_path_trg):
    tokenizer_trg = ByteLevelBPETokenizer()
    tokenizer_trg.train(files=[data_path_train_trg], vocab_size=opt.trg_vocab_size, min_frequency=2,
                        special_tokens=[
                            "<pad>",
                            "<mask>",
                            "<unk>",
                            "<s>",
                            "</s>",
                        ])
    if not os.path.exists(tokenizer_path_trg):
        os.makedirs(tokenizer_path_trg)
    tokenizer_trg.save_model(tokenizer_path_trg)
    return tokenizer_path_src, tokenizer_path_trg, data_path_train_src, data_path_train_trg


if __name__ == "__main__":

    opt = TrainOptions().parse()  # get training options
    opt.epoch = opt.current_epoch
    tokenizer_path_src, tokenizer_path_trg, data_path_train_src, data_path_train_trg = prepare_tokenizer(opt)
    data_path_eval_src = opt.data_path_eval_src.format_map(vars(opt))
    data_path_eval_trg = opt.data_path_eval_trg.format_map(vars(opt))
    print("tokenizer_path_src",tokenizer_path_src,"data_path_train_src",data_path_train_src,"data_path_eval_src",data_path_eval_src)
    print("tokenizer_path_trg",tokenizer_path_trg,"data_path_train_trg",data_path_train_trg,"data_path_eval_trg",data_path_eval_trg)

    step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy = load_trails(opt)

    train_dataset = ParallelCorpus(corpus_path_src=data_path_train_src, corpus_path_trg=data_path_train_trg,
                                   tokenizer_path_src=tokenizer_path_src, tokenizer_path_trg=tokenizer_path_trg)
    eval_dataset = ParallelCorpus(corpus_path_src=data_path_eval_src, corpus_path_trg=data_path_eval_trg,
                                  tokenizer_path_src=tokenizer_path_src, tokenizer_path_trg=tokenizer_path_trg)
    #     train_dataset = parallelCorpus(corpus_path_src=training_config["data_path_train_src"], corpus_path_trg=training_config["data_path_train_trg"]  , tokenizer_path_src=training_config["tokenizer_path_src"] , tokenizer_path_trg=training_config["tokenizer_path_trg"])
    #     eval_dataset = parallelCorpus(corpus_path_src=training_config["data_path_eval_src"], corpus_path_trg=training_config["data_path_eval_trg"]  , tokenizer_path_src=training_config["tokenizer_path_src"] , tokenizer_path_trg=training_config["tokenizer_path_trg"])

    dataloader_train = DataLoader(train_dataset, collate_fn=pad_to_max_with_mask, batch_size=opt.batch_size,
                                  shuffle=True)
    dataloader_eval = DataLoader(eval_dataset, collate_fn=pad_to_max_with_mask, batch_size=opt.batch_size,
                                 shuffle=False)
    opt.dataloader_length = len(dataloader_train)
    model = TransformerModel3(opt)  # create a dataset given opt.dataset_mode and other options
    model.setup()  # regular setup: load and print networks; create schedulers
    model.save_network()
    # model.load_network()
    # total_iters = 0                # the total number of training iterations
    for epoch in range(opt.num_of_epochs):
        opt.epoch = epoch
        loss_sum_train = 0
        err_train = 0
        num_tokens_train = 0

        #  train loop

        for i, batch in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
            batch = tuple(t.to(opt.device) for t in batch)
            # b_text_src, b_text_trg, b_mask_src, b_mask_trg = batch

            model.set_input(batch)
            loss_sum_eval_result, err_result, num_tokens_result = model.optimize_parameters()
            loss_sum_train += loss_sum_eval_result
            err_train += err_result
            num_tokens_train += num_tokens_result
            # label_smoothing(b_text_trg[:,1:], training_config["trg_vocab_size"], training_config["num_special_tokens_trg"])

            #  b_predicts = torch.argmax(b_outputs, dim=-1)
            #  correct += (b_predicts == b_labels).sum().item()

        train_loss = loss_sum_train / len(dataloader_train)
        train_losses.append(train_loss)
        train_acc = 1 - err_train / num_tokens_train
        train_accuracy.append(train_acc)

        loss_sum_eval = 0
        err_eval = 0
        num_tokens_eval = 0

        #  eval_loop
        for i, batch in tqdm(enumerate(dataloader_eval)):
            batch = tuple(t.to(opt.device) for t in batch)

            model.set_input(batch)
            loss_sum_eval_result, err_result, num_tokens_result = model.eval()
            loss_sum_eval += loss_sum_eval_result
            err_eval += err_result
            num_tokens_eval += num_tokens_result

        eval_loss = loss_sum_eval / len(dataloader_eval)
        eval_losses.append(eval_loss)
        eval_acc = 1 - err_eval / num_tokens_eval
        eval_accuracy.append(eval_acc)
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print(f'saving the model at the end of epoch{epoch}')
            # model.save_networks('latest')
            model.save_network()

        print(
            f'Epoch: {epoch + 1} \n Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f} \n Eval Loss: {eval_loss:.6f}, Eval Acc: {eval_acc:.6f}')
    opt.epoch = "latest"
    model.save_network()
    save_trails(opt, step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy)
