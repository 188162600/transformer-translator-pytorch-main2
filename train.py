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
from models.transformer_model import TransformerModel

from torch.optim import AdamW

from data_.parallel_corpus import ParallelCorpus
from data_.length_batch_sampler import LengthBatchSampler
import json
import os
import argparse
import utils.utils as utils
from torch.utils.data import random_split
from datasets import load_dataset
import os


def load_trails(opt):
    step_losses = list()
    step_losses_path = opt.step_losses_pth.format_map(vars(opt))
    if os.path.exists(step_losses_path):
        with open(step_losses_path, 'r') as file:
            step_losses = json.load(file)
            file.close()

    train_losses = list()
    train_losses_path = opt.train_losses_pth.format_map(vars(opt))
    if os.path.exists(opt.train_losses_pth):
        with open(train_losses_path, 'r') as file:
            train_losses = json.load(file)
            file.close()

    eval_losses = list()
    eval_losses_path = opt.eval_losses_pth.format_map(vars(opt))
    if os.path.exists(eval_losses_path):
        with open(eval_losses_path, 'r') as file:
            eval_losses = json.load(file)
            file.close()

    train_accuracy = list()
    train_accuracy_path = opt.train_accuracy_pth.format_map(vars(opt))
    if os.path.exists(train_accuracy_path):
        with open(train_accuracy_path, 'r') as file:
            train_accuracy = json.load(file)
            file.close()

    eval_accuracy = list()
    eval_accuracy_path = opt.eval_accuracy_pth.format_map(vars(opt))
    if os.path.exists(eval_accuracy_path):
        with open(eval_accuracy_path, 'r') as file:
            eval_accuracy = json.load(file)
            file.close()

    return step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy


def save_trails(opt, step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy):
    
    step_losses_path = opt.step_losses_pth.format_map(vars(opt))
  
    utils.mkdir(os.path.dirname(step_losses_path))
    with open(step_losses_path, 'w+') as file:
        json.dump(step_losses, file)
        file.close()
    train_losses_path = opt.train_losses_pth.format_map(vars(opt))
    utils.mkdir(os.path.dirname(train_losses_path))
    with open(train_losses_path, 'w+') as file:
        json.dump(train_losses, file)
        file.close()
    eval_losses_path = opt.eval_losses_pth.format_map(vars(opt))
    utils.mkdir(os.path.dirname(eval_losses_path))
    with open(eval_losses_path, 'w+') as file:
        json.dump(eval_losses, file)
        file.close()
    train_accuracy_path = opt.train_accuracy_pth.format_map(vars(opt))
    utils.mkdir(os.path.dirname(train_accuracy_path))
    with open(train_accuracy_path, 'w+') as file:
        json.dump(train_accuracy, file)
        file.close()
    eval_accuracy_path = opt.eval_accuracy_pth.format_map(vars(opt))
    utils.mkdir(os.path.dirname(eval_accuracy_path))
    with open(eval_accuracy_path, 'w+') as file:
        json.dump(eval_accuracy, file)
        file.close()


def prepare_tokenizer(opt):
    tokenizer_path_src = opt.tokenizer_path_src.format_map(vars(opt))
    tokenizer_path_trg = opt.tokenizer_path_trg.format_map(vars(opt))
    os.makedirs(os.path.dirname(tokenizer_path_src), exist_ok=True)
    os.makedirs(os.path.dirname(tokenizer_path_trg), exist_ok=True)
    data_path_src = opt.tokenizer_data_source_src.format_map(vars(opt))
    data_path_trg = opt.tokenizer_data_source_trg.format_map(vars(opt))

    if opt.retokenize or not os.path.exists(tokenizer_path_src):

        from tokenizers.implementations import ByteLevelBPETokenizer

        tokenizer_src = ByteLevelBPETokenizer()
        tokenizer_src.train(files=[data_path_src], vocab_size=opt.src_vocab_size, min_frequency=2,
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
    if opt.retokenize or not os.path.exists(tokenizer_path_trg):
        
            from tokenizers.implementations import ByteLevelBPETokenizer
        #if os.path.exists(tokenizer_path_trg) and os.path.exists(data_path_train_trg):
            tokenizer_trg = ByteLevelBPETokenizer()
            tokenizer_trg.train(files=[data_path_trg], vocab_size=opt.trg_vocab_size, min_frequency=2,
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
    return tokenizer_path_src, tokenizer_path_trg
def get_data_loader(opt):
    
    tokenizer_path_src, tokenizer_path_trg = prepare_tokenizer(opt)
    data_path_train = opt.data_path_train.format_map(vars(opt))
    data_path_eval = opt.data_path_eval.format_map(vars(opt))
    
   
    
    
    train_dataset = ParallelCorpus(data_path_train,
                                   tokenizer_path_src=tokenizer_path_src, tokenizer_path_trg=tokenizer_path_trg)
    eval_dataset =ParallelCorpus(data_path_eval,
                                  tokenizer_path_src=tokenizer_path_src, tokenizer_path_trg=tokenizer_path_trg)
    #print("calculating the max length of the dataset")
    print("train_dataset length",len(train_dataset))
    print("eval_dataset length",len(eval_dataset))
    train_seq_length =train_dataset.get_max_line_length()
    eval_seq_length = eval_dataset.get_max_line_length()
    print("max length of the train dataset",train_seq_length,"max length of the eval dataset",eval_seq_length)
    opt.max_src_seq_length = max(train_seq_length, eval_seq_length)
    filter_len=max(opt.batch_ignore_len,opt.batch_max_len)
    opt.max_src_seq_length = min(opt.max_src_seq_length,filter_len)
   
    train_sampler= LengthBatchSampler(train_dataset.get_lines_length(),max_len=opt.batch_max_len,ignore_len=opt.batch_ignore_len,max_batch=opt.max_batch, shuffle=True)
    eval_sampler = LengthBatchSampler(eval_dataset.get_lines_length(),max_len=1,ignore_len=opt.batch_ignore_len,max_batch =opt.max_batch,shuffle=False)
   
    #print("train_sampler",list(train_sampler)[0:5])
    #print("eval_sampler",list(eval_sampler)[0:5])
    dataloader_train= DataLoader(train_dataset, collate_fn=pad_to_max_with_mask, batch_sampler=train_sampler)
    dataloader_eval = DataLoader(eval_dataset, collate_fn=pad_to_max_with_mask, batch_sampler=eval_sampler)
    if opt.save_in_memory:
        train_dataset= list(tqdm(dataloader_train,desc="loading train dataset into memory",total=len(dataloader_train)))
        eval_dataset = list(tqdm(dataloader_eval,desc="loading eval dataset into memory",total=len(dataloader_train)))
        dataloader_train= DataLoader(train_dataset, batch_size=None,shuffle=True)
        dataloader_eval = DataLoader(eval_dataset, batch_size=None,shuffle=False)
        
        
    return dataloader_train, dataloader_eval



if __name__ == "__main__":

    opt = TrainOptions().parse()  # get training options
    opt.epoch = opt.current_epoch
    tokenizer_path_src, tokenizer_path_trg = prepare_tokenizer(opt)
    # data_path_eval_src = opt.data_path_eval_src.format_map(vars(opt))
    # data_path_eval_trg = opt.data_path_eval_trg.format_map(vars(opt))

    step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy = load_trails(opt)

    
    dataloader_train,dataloader_eval = get_data_loader(opt)
    print("dataloader_train length",len(dataloader_train))
    print("dataloader_eval length",len(dataloader_eval))
            
    
    opt.dataloader_length = len(dataloader_train)
    model = TransformerModel(opt)  # create a dataset given opt.dataset_mode and other options
    model.setup()  # regular setup: load and print networks; create schedulers
    model.save_network()
    save_trails(opt, step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy)
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
            b_text_src, b_text_trg, b_mask_src, b_mask_trg = batch
            #print(b_text_src.device, b_text_trg.device, b_mask_src.device, b_mask_trg.device)

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
        bleu_score = 0
        num_batch=0

        #  eval_loop
        for i, batch in tqdm(enumerate(dataloader_eval), total=len(dataloader_eval)):
            batch = tuple(t.to(opt.device) for t in batch)

            model.set_input(batch)
            loss_sum_eval_result, err_result, num_tokens_result,blue_eval_result,num_batch_eval = model.eval()
            #print("batch_eval",batch_eval)
            loss_sum_eval += loss_sum_eval_result
            err_eval += err_result
            num_tokens_eval += num_tokens_result
            bleu_score += blue_eval_result
            num_batch+=num_batch_eval

        eval_loss = loss_sum_eval / len(dataloader_eval)
        eval_losses.append(eval_loss)
        eval_acc = 1 - err_eval / num_tokens_eval
        eval_accuracy.append(eval_acc)
        eval_bleu = bleu_score / len(dataloader_eval)
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print(f'saving the model at the end of epoch{epoch}')
            # model.save_networks('latest')
            model.save_network()

        print(
            f'Epoch: {epoch + 1} \n Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f} \n Eval Loss: {eval_loss:.6f}, Eval Acc: {eval_acc:.6f}, Eval Bleu: {eval_bleu:.6f} \n')
        save_trails(opt, step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy)

    opt.epoch = "latest"
    model.save_network()
    save_trails(opt, step_losses, train_losses, eval_losses, train_accuracy, eval_accuracy)
