import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers.implementations import ByteLevelBPETokenizer
import csv
import re
import copy
import os


class ParallelCorpus(Dataset):
    '''
    read the parallel corpus 
    return src text and trg text (encoded, surely)
    '''

    def __init__(self, corpus_path_src, corpus_path_trg, tokenizer_path_src, tokenizer_path_trg,device):
        text_src = list()
        text_trg = list()
        os.makedirs(tokenizer_path_src, exist_ok=True)
        os.makedirs(tokenizer_path_trg, exist_ok=True)

        tokenizer_src = ByteLevelBPETokenizer(
                tokenizer_path_src + "/vocab.json",
                tokenizer_path_src + "/merges.txt"
                )
        tokenizer_trg = ByteLevelBPETokenizer(
                tokenizer_path_trg + "/vocab.json",
                tokenizer_path_trg + "/merges.txt"
                )
        print("tokenizer loaded",tokenizer_path_src,tokenizer_path_trg)
        
        '''
        handle the src
        '''
        with open(corpus_path_src, 'r',encoding='UTF-8') as file:
            line = file.readline()

            while line:
                line = line.strip()
                encoding = tokenizer_src.encode(line)
                token_list = [3] + encoding.ids + [4]

                text_src.append(token_list)
                #  mask_src.append(encoding.)

                line = file.readline()
        print("src done")
        
        '''
        handle the trg
        '''
        with open(corpus_path_trg, 'r',encoding='UTF-8') as file:
            line = file.readline()

            while line:
                line = line.strip()
                encoding = tokenizer_trg.encode(line)
                token_list = [3] + encoding.ids + [4]

                text_trg.append(token_list)
                #  mask_trg.append(encoding.)

                line = file.readline()
        print("trg done")

        self.text_src = text_src
        self.text_trg = text_trg

        assert(len(self.text_src) == len(self.text_trg))

    
    def __len__(self):
        return len(self.text_src)


    def __getitem__(self, idx):
        #  tensorlize
        text_src_tensor = torch.tensor(self.text_src[idx], dtype=torch.long)
        text_trg_tensor = torch.tensor(self.text_trg[idx], dtype=torch.long)

        
        return text_src_tensor, text_trg_tensor
