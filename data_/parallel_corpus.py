import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_from_disk
from tokenizers.implementations import ByteLevelBPETokenizer
import csv
import re
import copy
import os
import tqdm

class ParallelCorpus:
    '''
    read the parallel corpus 
    return src text and trg text (encoded, surely)
    '''

    def __init__(self, path, tokenizer_path_src, tokenizer_path_trg):
      
     
        self.tokenizer_src = ByteLevelBPETokenizer(
                tokenizer_path_src + "/vocab.json",
                tokenizer_path_src + "/merges.txt"
                )
        self.tokenizer_trg = ByteLevelBPETokenizer(
                tokenizer_path_trg + "/vocab.json",
                tokenizer_path_trg + "/merges.txt"
                )
        self.dataset=load_from_disk(path)
        @staticmethod
        def get_length(text):
            return max(map(len,  text["translation"].values()))+2
        self.lines_length=torch.zeros(len(self.dataset),dtype=torch.long)
        for i,data in tqdm.tqdm(enumerate( self.dataset),desc="Compute Lines Length",total=len(self.dataset)):
            self.lines_length[i]=get_length(data)
        self.max_line_length=self.lines_length.max().item()
          
        # print("tokenizer loaded",tokenizer_path_src,tokenizer_path_trg)
        
        # '''
        # handle the src
        # '''
        # with open(corpus_path_src, 'r',encoding='UTF-8') as file:
        #     line = file.readline()

        #     while line:
        #         line = line.strip()
        #         encoding = tokenizer_src.encode(line)
        #         token_list = [3] + encoding.ids + [4]

        #         text_src.append(token_list)
        #         #  mask_src.append(encoding.)

        #         line = file.readline()
        # print("src done")
        
        # '''
        # handle the trg
        # '''
        # with open(corpus_path_trg, 'r',encoding='UTF-8') as file:
        #     line = file.readline()

        #     while line:
        #         line = line.strip()
        #         encoding = tokenizer_trg.encode(line)
        #         token_list = [3] + encoding.ids + [4]

        #         text_trg.append(token_list)
        #         #  mask_trg.append(encoding.)

        #         line = file.readline()
        # print("trg done")

        # self.text_src = text_src
        # self.text_trg = text_trg

        # assert(len(self.text_src) == len(self.text_trg))

    def __iter__(self):
        for line in self.dataset:
            line=line["translation"]
            src_text,trg_text =line.values()
            src_encoding=self.tokenizer_src.encode(src_text)
            trg_encoding=self.tokenizer_trg.encode(trg_text)
            src = [3] + src_encoding.ids + [4]
            trg = [3] + trg_encoding.ids + [4]
            text_src_tensor = torch.tensor(src, dtype=torch.long)
            text_trg_tensor = torch.tensor(trg, dtype=torch.long)
            yield text_src_tensor, text_trg_tensor
            
    def __len__(self):
        return len(self.dataset)

    def get_max_line_length(self):
        return self.max_line_length
    def get_lines_length(self):
        return self.lines_length
    
    
    def __getitem__(self, idx):
        #  tensorlize
        #print("idx",idx)
        line=self.dataset[idx]["translation"]
        src_text,trg_text =line.values()     
        src_encoding=self.tokenizer_src.encode(src_text)
        trg_encoding=self.tokenizer_trg.encode(trg_text)
        src = [3] + src_encoding.ids + [4]
        trg = [3] + trg_encoding.ids + [4]
        
        text_src_tensor = torch.tensor(src, dtype=torch.long)
        text_trg_tensor = torch.tensor(trg, dtype=torch.long)

        
        return text_src_tensor, text_trg_tensor
    