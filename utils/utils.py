import torch
import torch.nn.functional as F

import os
from torch.optim.lr_scheduler import LambdaLR

from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction




def pad_to_max_with_mask(data):
    """
    data: list of tuples (text_src_tensor, text_trg_tensor)
    
    return (b_text_src_tensor, b_text_trg_tensor, b_mask_src_tensor, b_mask_trg_tensor)
    """
    max_len_src = max(text_tuple[0].shape[-1] for text_tuple in data)
    max_len_trg = max(text_tuple[1].shape[-1] for text_tuple in data)
    sum_len_src = sum(text_tuple[0].shape[-1] for text_tuple in data)
    sum_len_trg = sum(text_tuple[1].shape[-1] for text_tuple in data)
    
    #print("max_len_src",max_len_src,"max_len_trg",max_len_trg,"sum_len_src",sum_len_src,"sum_len_trg",sum_len_trg,"len(data)",len(data),"max",len(data)*max(max_len_src,max_len_trg))
    #  create the mask
    b_text_src_tensor = torch.zeros((len(data), max_len_src), dtype = torch.long)
    b_text_trg_tensor = torch.zeros((len(data), max_len_trg), dtype = torch.long)
    b_mask_src_tensor = torch.ones_like(b_text_src_tensor, dtype = torch.bool)
    b_mask_trg_tensor = torch.ones_like(b_text_trg_tensor, dtype = torch.bool,)

    for i in range(len(data)):
        b_text_src_tensor[i] = F.pad(data[i][0], (0, max_len_src - data[i][0].shape[-1]), "constant", 0)
        b_text_trg_tensor[i] = F.pad(data[i][1], (0, max_len_trg - data[i][1].shape[-1]), "constant", 0)
        b_mask_src_tensor[i, :data[i][0].shape[-1]] = 0
        b_mask_trg_tensor[i, :data[i][1].shape[-1]] = 0
    return b_text_src_tensor, b_text_trg_tensor, b_mask_src_tensor, b_mask_trg_tensor




def label_smoothing(tensor, num_classes, num_special_tokens, smoothing_value = 0.1):
    """
    """

    smooth_label = F.one_hot(tensor, num_classes=num_classes).to(torch.float)

    smooth_label[smooth_label == 0] = smoothing_value / (num_classes - num_special_tokens - 1 )
    smooth_label[smooth_label == 1] = 1 - smoothing_value


    return smooth_label




def greedy_decoding(model, text_src, max_output_len = 100, BOS_id = 3, EOS_id = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    text_trg = torch.tensor([[BOS_id]]).to(device)

    while text_trg.shape[-1] < max_output_len:
        #  text_trg = F.pad(text_trg, (0, 1), "constant", 0)
        mask_src = torch.zeros_like(text_src, dtype = torch.bool).to(device)
        mask_trg = torch.zeros_like(text_trg, dtype = torch.bool).to(device)

        predicted_log_distributions = model(text_src, text_trg, mask_src, mask_trg)
        next_log_prediction = predicted_log_distributions[:,-1:,:]
        next_prediction = torch.argmax(next_log_prediction, dim=2)


        #  text_trg[0][-1] = next_prediction
        #  text_trg = F.pad(text_trg, (0, 1), "constant", next_prediction)
        text_trg = torch.cat((text_trg, next_prediction), dim = 1)

        if next_prediction == EOS_id:
            break

    return text_trg

import copy
from typing import Optional, Any, Union, Callable

import torch
import warnings
from torch import Tensor

def get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]
def detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

def generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
def get_causal_mask(size,device):
   
    raw_mask = torch.ones((size, size), dtype=torch.bool)
    triled_mask = torch.tril(raw_mask)
    triled_mask = ~triled_mask
    return triled_mask.to(device)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def clear_for_bleu(data: torch.Tensor, eos=4) -> list:
    data_list = data.tolist()
    if eos in data_list:
        return data_list[:data_list.index(eos)]
    return data_list
def select_smoothing_function(sentence_length):
    """
    Selects a smoothing function based on the length of the candidate sentence, with special handling for very short sentences.

    :param sentence_length: The length of the candidate sentence.
    :return: A smoothing function from NLTK's SmoothingFunction.
    """
    cc = SmoothingFunction()

    if sentence_length <= 3:
        # Very short sentences: Use the most aggressive smoothing method
        return cc.method7
    elif 4 <= sentence_length < 10:
        # Short to medium length sentences: Use a balanced smoothing method
        return cc.method4
    else:
        # Longer sentences: Less aggressive smoothing, assuming more n-gram matches
        return cc.method1

    
smoothie = SmoothingFunction()
def get_bleu_score(reference: torch.Tensor, candidate: torch.Tensor, eos=4,
                   methods=(smoothie.method1,smoothie.method2,smoothie.method3,smoothie.method4,smoothie.method5,smoothie.method7)):
    assert reference.size(0) == candidate.size(0), "Reference and candidate must have the same batch size"

    batch_size = reference.size(0)

    scores=[0]*(len(methods)+1)
    for ref, cand in zip(reference, candidate):
        ref_clean = clear_for_bleu(ref, eos)
        cand_clean = clear_for_bleu(cand, eos)
        # Debugging print statements
        #print(ref_clean,cand_clean)
        for i,method in enumerate(methods):
            scores[i]+=sentence_bleu([ref_clean],cand_clean,smoothing_function=method)
        scores[-1]+=sentence_bleu([ref_clean],cand_clean,smoothing_function=select_smoothing_function(min(len(ref_clean),len(cand_clean))))



    return torch.tensor(scores), batch_size


