##############################
## BERT - Commentary Tokenizer
##############################

import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from transformers import BertTokenizer


def get_len_of_bigger_sentence(sentences, default_max_len=None):
    """Bert demands a default maximun phrase len to work properly.
    This lenght is measured by number of words. If the phrase is
    smaller than maximun len, the remaining blank elements are filled
    with padding tokens.
    """
    if default_max_len:
      return default_max_len

    max_len = 0
    for sent in sentences:
        max_len = max(max_len, len(sent.split(' ')))
  
    print('Max sentence length: ', max_len)
    return max_len


def tokenize_sentences(sentences, max_length, tokenizer):
    """Tokenize all of the sentences and map the tokens to their word IDs.
    """
    tokenized_sentences = []
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
        tokenized_sentences.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    tokenized_sentences = torch.cat(tokenized_sentences, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return (tokenized_sentences, attention_masks)


def tokenize_dataset(sentences, labels, max_phrase_len=None):
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
        tokenized_sentences(tensor): bert encoded phrases
        attention_masks(tensor): tensor containing each phrase's own
                                 attention mask
        labels(tensor): each row's label, stored as a tensor
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_phrase_len = get_len_of_bigger_sentence(sentences, max_phrase_len)
    tokenized_sentences, attention_masks = tokenize_sentences(sentences, max_phrase_len, tokenizer)
    labes_tensor_obj = torch.tensor(labels)
    return (tokenized_sentences, attention_masks, labes_tensor_obj)
