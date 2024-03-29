from numpy import shape
import torch
from torch.cuda import utilization
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tokenizers.implementations import ByteLevelBPETokenizer

from src.dataset import parallelCorpus
from src.utils import greedy_decoding, pad_to_max_with_mask, label_smoothing
from src.model import Transformer
import json
import os
import argparse


def load_model(model_path_src, translation_config):
    '''
    load model
    '''
    model = Transformer(
            model_dimension     = translation_config["model_dimension"],
            src_vocab_size      = translation_config["src_vocab_size"],
            trg_vocab_size      = translation_config["trg_vocab_size"],
            number_of_heads     = translation_config["number_of_heads"],
            number_of_layers    = translation_config["number_of_layers"],
            dropout_probability = translation_config["dropout_probability"],
            )


    if os.path.exists(model_path_src):
        model_dict = torch.load(translation_config['model_path_src'])
        model.load_state_dict(model_dict)

    return model


def translate(translation_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    '''
    model
    load model 
    '''
    model = load_model(translation_config['model_path_src'], translation_config).to(device)

    '''
    dataloader
    '''
    
    test_dataset = parallelCorpus(corpus_path_src=translation_config["data_path_test_src"], corpus_path_trg=translation_config["data_path_test_trg"]  , tokenizer_path_src=translation_config["tokenizer_path_src"] , tokenizer_path_trg=translation_config["tokenizer_path_trg"])
    dataloader_test = DataLoader(test_dataset, collate_fn=pad_to_max_with_mask, batch_size=5, shuffle=False)


    '''
    creterian
    '''
    creterian = nn.KLDivLoss(reduction='batchmean')

        

    '''
    calculate the loss and acc
    '''
    loss_sum_test = 0
    err            = 0
    num_tokens     = 0

    #  get loss and acc
    for i, batch in enumerate(dataloader_test):
        batch = tuple(t.to(device) for t in batch)
        b_text_src, b_text_trg, b_mask_src, b_mask_trg = batch

        with torch.no_grad():
            #  b_mask_src = torch.ones_like(b_text_src, dtype = torch.bool)
            #  b_text_src = torch.ones_like(b_text_src, dtype = torch.long)
            b_predicted_log_distributions = model(b_text_src, b_text_trg[:,:-1], b_mask_src, b_mask_trg[:,:-1])
            b_smooth_label = label_smoothing(b_text_trg[:,1:], translation_config["trg_vocab_size"], translation_config["num_special_tokens_trg"])

            loss = creterian(b_predicted_log_distributions, b_smooth_label)
            loss_scalar = loss.item()
            loss_sum_test += loss_scalar

            b_predictions = torch.argmax(b_predicted_log_distributions, dim=2)
            err += (b_predictions != b_text_trg[:,1:]).sum().item()
            num_tokens += torch.sum(~b_mask_trg[:,:-1]).item()


    test_loss = loss_sum_test / len(dataloader_test)
    test_acc = 1 - err / num_tokens

    print(f'Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}')


    for i, batch in enumerate(dataloader_test):
        batch = tuple(t.to(device) for t in batch)
        b_text_src, b_text_trg, b_mask_src, b_mask_trg = batch

        with torch.no_grad():
            b_mask_src = torch.ones_like(b_text_src, dtype = torch.bool)
            #  b_text_src = torch.ones_like(b_text_src, dtype = torch.long)
            b_predicted_log_distributions = model(b_text_src, b_text_trg[:,:-1], b_mask_src, b_mask_trg[:,:-1])
            b_smooth_label = label_smoothing(b_text_trg[:,1:], translation_config["trg_vocab_size"], translation_config["num_special_tokens_trg"])

            loss = creterian(b_predicted_log_distributions, b_smooth_label)
            loss_scalar = loss.item()
            loss_sum_test += loss_scalar

            b_predictions = torch.argmax(b_predicted_log_distributions, dim=2)
            err += (b_predictions != b_text_trg[:,1:]).sum().item()
            num_tokens += torch.sum(~b_mask_trg[:,:-1]).item()


    test_loss = loss_sum_test / len(dataloader_test)
    test_acc = 1 - err / num_tokens

    print("when the src is masked out")
    print(f'Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.6f}')


    '''
    translate and output to another file
    '''
    dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

    tokenizer_path_trg = translation_config["tokenizer_path_trg"]

    tokenizer = ByteLevelBPETokenizer(
            tokenizer_path_trg + "/vocab.json",
            tokenizer_path_trg + "/merges.txt"
            )

    with open(translation_config['translation_output_path'], 'w') as file:
        for i, batch in enumerate(dataloader_test):
            batch = tuple(t.to(device) for t in batch)
            text_src, text_trg = batch

            trg_out = greedy_decoding(model, text_src, max_output_len=translation_config["max_target_tokens"]).squeeze(0).tolist()

            file.write(tokenizer.decode(trg_out) + '\n')
        file.close()




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--translation_output_path"    , type=str   , help="the file to write the result"                , default='./test.out')
    parser.add_argument("--dropout_probability"    , type=float , help="dropout prob"                                        , default=1e-1)
    parser.add_argument("--src_vocab_size"         , type=int   , help="src vocab size"                                      , default=5000)
    parser.add_argument("--trg_vocab_size"         , type=int   , help="trg vocab size"                                      , default=5000)
    parser.add_argument("--model_dimension"        , type=int   , help="model dimmention"                                    , default=512)
    parser.add_argument("--number_of_heads"        , type=int   , help="number of heads"                                     , default=8)
    parser.add_argument("--number_of_layers"       , type=int   , help="number of layers"                                    , default=6)
    parser.add_argument("--num_special_tokens_trg" , type=int   , help="number of special tokens of target"                  , default=2)
    parser.add_argument("--max_target_tokens"        , type=int   , help="max tokens for the output"                                     , default=100)
    parser.add_argument("--tokenizer_path_src"     , type=str   , help="the saved tokenizer of src language"                 , default='./data/tokenizer_zh')
    parser.add_argument("--tokenizer_path_trg"     , type=str   , help="the saved tokenizer of trg language"                 , default='./data/tokenizer_en')
    parser.add_argument("--data_path_test_src"    , type=str   , help="the testing dataset of src language"                , default='./data/test.zh')
    parser.add_argument("--data_path_test_trg"    , type=str   , help="the testing dataset of trg language"                , default='./data/test.en')
    parser.add_argument("--model_path_dst"         , type=str   , help="the directory to save model"                         , default='./saved_models/saved_dict.pth')
    parser.add_argument("--model_path_src"         , type=str   , help="the directory to load model"                         , default='./saved_models/saved_dict.pth')

    
    args = parser.parse_args()

    translation_config = dict()
    for arg in vars(args):
        translation_config[arg] = getattr(args, arg)

    translate(translation_config)
