import torch
import os
import json

from typing import Union
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer

def bert_encode(text, max_len, tokenizer):
    token_idx = torch.tensor(tokenizer.encode(text, max_length=max_len, add_special_tokens=True,
                                             padding='max_length', truncation=True))
    mask_token_id = tokenizer.pad_token_id
    mask = torch.zeros(token_idx.shape)

    for i, tokens in enumerate(token_idx):
        if tokens != mask_token_id:
            mask[i] = 1

    return token_idx, mask

def preprocess_gossip_cop_dataset(item, max_len, tokenizer):
    news_token_ids,news_mask = bert_encode(item['news'], max_len, tokenizer)
    good_explain_token_ids,good_explain_mask = bert_encode(item['the_good'], max_len, tokenizer)
    bad_explain_token_ids,bad_explain_mask = bert_encode(item['the_bad'], max_len, tokenizer)
    ugly_explain_token_ids,ugly_explain_mask = bert_encode(item['the_ugly'], max_len, tokenizer)

    data_item ={
        "news_token_ids":news_token_ids,
        "news_mask":news_mask,
        "label": item['label'],
        "good_explain_token_ids":good_explain_token_ids,
        "good_explain_mask":good_explain_mask,
        "bad_explain_token_ids":bad_explain_token_ids,
        "bad_explain_mask":bad_explain_mask,
        "ugly_explain_token_ids":ugly_explain_token_ids,
        "ugly_explain_mask":ugly_explain_mask,
    }

    return data_item

class GossipCopDataset(Dataset):
    """EGN-paper datasets."""
    def __init__(self, data_path:Union[str, Path], mode:str, slm_name, max_len:int):
        """
        :param data_path: path to the folder with data files.
        :param mode: "train", "val", "test"
        :param slm_name: the embedding / small language model name
        :param max_len: max length of the input sequence
        """
        self._mode = mode
        self._data_path = os.path.join(data_path, mode)

        self._tokenizer = BertTokenizer.from_pretrained(slm_name)
        self._max_len = max_len

    def __len__(self):
        return len(os.listdir(self._data_path))

    def __getitem__(self, idx):

        file_path = os.path.join(self._data_path, f"{idx}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return preprocess_gossip_cop_dataset(data, self._max_len, self._tokenizer)