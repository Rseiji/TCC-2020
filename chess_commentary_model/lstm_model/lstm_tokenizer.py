import pandas as pd
import numpy as np
import bert
import math

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers


def get_tokenizer():
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    link = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    bert_layer = hub.KerasLayer(link, trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    return tokenizer


def tokenize_dataset(df, tokenizer, batch_size=32, seed=0, cross_val_proportion = 0.1):
    assert (not df['comment'].isna().any()), "Rows without comment are not allowed!"
    reviews = _preprocessed_dataset(df)
    tokenized_reviews = _tokenize_dataset(reviews, tokenizer)
    df['tokenized_commentary'] = tokenized_reviews
    return df


def _preprocessed_dataset(df):
    return df['comment'].tolist()


def _tokenize_dataset(reviews, tokenizer):
    tokenized_reviews = [_tokenize_reviews(review, tokenizer) for review in reviews]
    return tokenized_reviews


def _tokenize_reviews(text_reviews, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))

