##############################
## BERT - Commentary Tokenizer
##############################

import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from transformers import BertTokenizer


def get_phrases_maximun_len(sentences, tokenizer, default_max_len=None):
    """Bert demands a default maximun phrase len to work properly.
    This lenght is measured by number of words. If the phrase is
    smaller than maximun len, the remaining blank elements are filled
    with padding tokens.
    """
    if default_max_len:
      return default_max_len

    max_len = 0
    for sent in sentences:
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
  
    print('Max sentence length: ', max_len)
    return max_len


def tokenize_sentences(sentences, max_length, tokenizer):
    """Tokenize all of the sentences and map the tokens to their word IDs.
    """
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True,
                            max_length = max_length,
                            truncation=True,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt'
                      )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return (input_ids, attention_masks)


def make_dataset(sentences, labels, max_phrase_len=None):
    """Makes a tuple containing encoded input phrases, its corresponding
    attention masks and maximum phrase length.
    Args:
        sentences(list): list of strings, with phrases to be encoded
        labels(list): sentences list's labels: list elements must be either
        0 or 1
        max_phrase_len(int): maximum phrase length to be considered by BERT
        tokenizer. If None, will be defined as maximum phrase lenght in
        sentences list.
    Returns:
        input_ids(tensor): bert encoded phrases
        attention_masks(tensor): tensor containing each phrase's own
                                 attention mask
        labels(tensor): each row's label, stored as a tensor
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_phrase_len = get_phrases_maximun_len(sentences, tokenizer, max_phrase_len)
    input_ids, attention_masks = tokenize_sentences(sentences, max_phrase_len, tokenizer)
    labels = torch.tensor(labels)
    return (input_ids, attention_masks, labels)
