from datasets import inspect_dataset, load_dataset_builder
import datasets
import  transformers
import  tokenizers.implementations
import  transformers
import argparse
from tqdm import tqdm
parser= argparse.ArgumentParser()

parser.add_argument("--src", type=str, help="source language", default="de")
parser.add_argument("--trg", type=str, help="target language", default="en")

args = parser.parse_args()

tokenizers.implementations.ByteLevelBPETokenizer()

inspect_dataset("wmt14", "scripts")
path=f"./datasets/{args.src}_to_{args.trg}"
builder = load_dataset_builder(
    "./scripts/wmt_utils.py",
    language_pair=(args.src, args.trg),
    subsets={
        datasets.Split.TRAIN:["europarl_v7",
                "commoncrawl",
                "multiun",
                "newscommentary_v9",
                "gigafren",
                "czeng_10",
                "yandexcorpus",
                "wikiheadlines_hi",
                "wikiheadlines_ru",
                "hindencorp_01",],
        datasets.Split.VALIDATION: ["newsdev2014", "newstest2013"],
        datasets.Split.TEST: ["newstest2014"]
       # datasets.Split.VALIDATION: ["euelections_dev2019"],
    },trust_remote_code=True

)
builder.download_and_prepare(trust_remote_code=True)

dataset=builder.as_dataset()
dataset.save_to_disk(path)
names="train","test","validation"
for name in names:
    src_file = open(f"{path}/{name}/{args.src}_to_{args.trg}.{args.src}", "w+",encoding="utf-8")
    trg_file = open(f"{path}/{name}/{args.src}_to_{args.trg}.{args.trg}", "w+",encoding="utf-8")
    for src_trg in tqdm( dataset[name],desc=f"writing {name}"):
        src_file.write(src_trg["translation"]["de"] + "\n")
        trg_file.write(src_trg["translation"]["en"] + "\n")
    src_file.close()
    trg_file.close()


# from datasets import inspect_dataset, load_dataset_builder
# import datasets
# import  transformers
# import  tokenizers.implementations
# import  transformers
# import argparse
# import os
# from tqdm import tqdm

# def download_and_prepare(opt):

#     inspect_dataset("wmt14", "scripts")
#     path=f"./datasets/{args.src}_to_{args.trg}"
#     builder = load_dataset_builder(
#         "./scripts/wmt_utils.py",
#         language_pair=(args.src, args.trg),
#         subsets={
#             datasets.Split.TRAIN:["europarl_v7",
#                     "commoncrawl",
#                     "multiun",
#                     "newscommentary_v9",
#                     "gigafren",
#                     "czeng_10",
#                     "yandexcorpus",
#                     "wikiheadlines_hi",
#                     "wikiheadlines_ru",
#                     "hindencorp_01",],
#             datasets.Split.VALIDATION: ["newsdev2014", "newstest2013"],
#             datasets.Split.TEST: ["newstest2014"]
#         # datasets.Split.VALIDATION: ["euelections_dev2019"],
#         },
#     )
#     builder.download_and_prepare()

#     dataset=builder.as_dataset()
#     dataset.save_to_disk(path)
#     names="train","test","validation"

    
#     for name in names:
#         src_file = open(f"{path}/{name}/{args.src}_to_{args.trg}.{args.src}", "w+",encoding="utf-8")
#         trg_file = open(f"{path}/{name}/{args.src}_to_{args.trg}.{args.trg}", "w+",encoding="utf-8")
#         for src_trg in tqdm( dataset[name],desc=f"writing {name}"):
#             src_file.write(src_trg["translation"]["de"] + "\n")
#             trg_file.write(src_trg["translation"]["en"] + "\n")
#         src_file.close()
#         trg_file.close()

# def prepare_tokenizer(opt):
#     tokenizer_path_src = opt.tokenizer_path_src.format_map(vars(opt))
#     tokenizer_path_trg = opt.tokenizer_path_trg.format_map(vars(opt))
#     os.makedirs(os.path.dirname(tokenizer_path_src), exist_ok=True)
#     os.makedirs(os.path.dirname(tokenizer_path_trg), exist_ok=True)
#     data_path_train_src = opt.data_path_train_src.format_map(vars(opt))
#     data_path_train_trg = opt.data_path_train_trg.format_map(vars(opt))

#     if opt.retokenize or not os.path.exists(tokenizer_path_src):

#         from tokenizers.implementations import ByteLevelBPETokenizer

#         tokenizer_src = ByteLevelBPETokenizer()
#         tokenizer_src.train(files=[data_path_train_src], vocab_size=opt.src_vocab_size, min_frequency=2,
#                             special_tokens=[
#                                 "<pad>",
#                                 "<mask>",
#                                 "<unk>",
#                                 "<s>",
#                                 "</s>",
#                             ])
#         if not os.path.exists(tokenizer_path_src):
#             os.makedirs(tokenizer_path_src)
#         tokenizer_src.save_model(tokenizer_path_src)
#     if opt.retokenize or not os.path.exists(tokenizer_path_trg):
        
#             from tokenizers.implementations import ByteLevelBPETokenizer
#         #if os.path.exists(tokenizer_path_trg) and os.path.exists(data_path_train_trg):
#             tokenizer_trg = ByteLevelBPETokenizer()
#             tokenizer_trg.train(files=[data_path_train_trg], vocab_size=opt.trg_vocab_size, min_frequency=2,
#                                 special_tokens=[
#                                     "<pad>",
#                                     "<mask>",
#                                     "<unk>",
#                                     "<s>",
#                                     "</s>",
#                                 ])
#             if not os.path.exists(tokenizer_path_trg):
#                 os.makedirs(tokenizer_path_trg)
                
#             tokenizer_trg.save_model(tokenizer_path_trg)
#     return tokenizer_path_src, tokenizer_path_trg, data_path_train_src, data_path_train_trg

# parser= argparse.ArgumentParser()

# parser.add_argument("--src", type=str, help="source language", default="de")
# parser.add_argument("--trg", type=str, help="target language", default="en")
# parser.add_argument("--tokenizer_path_src", type=str, help="the saved tokenizer of src language", default='./datasets/{src}_to_{trg}/tokenizer.src')
# parser.add_argument("--tokenizer_path_trg", type=str, help="the saved tokenizer of trg language", default='./datasets/{src}_to_{trg}/tokenizer.trg')
# parser.add_argument("--data_path_train_src", type=str, help="the training dataset of src language", default='./datasets/{src}_to_{trg}/train/{src}_to_{trg}.{src}')
# parser.add_argument("--data_path_train_trg", type=str, help="the training dataset of trg language", default='./datasets/{src}_to_{trg}/train/{src}_to_{trg}.{trg}')
# parser.add_argument("--data_path_eval_src", type=str, help="the eval dataset of src language", default='./datasets/{src}_to_{trg}/validation/{src}_to_{trg}.{src}')
# parser.add_argument("--data_path_eval_trg", type=str, help="the eval dataset of trg language", default='./datasets/{src}_to_{trg}/validation/{src}_to_{trg}.{trg}')
# parser.add_argument("--data_path_test_src", type=str, help="the testing dataset of src language", default='./datasets/{src}_to_{trg}/test/{src}_to_{trg}.{src}')
# parser.add_argument("--data_path_test_trg", type=str, help="the testing dataset of trg language", default='./datasets/{src}_to_{trg}/test/{src}_to_{trg}.{trg}')

# parser.add_argument("--retokenize",type=bool,default=False)
# opt = parser.parse_args()

# download_and_prepare(opt)
# prepare_tokenizer(opt)
