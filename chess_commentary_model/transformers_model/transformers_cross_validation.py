###################################
## BERT - Model Cross-validation
## Stratified K-fold implementation
###################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import time
import datetime
import io
import psutil
import humanize
import os
import GPUtil as GPU
import gc

from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import transformers_model_training as tmt
import bert_encoder


def cross_validation_stratified_k_fold(df, epochs, device, n_splits=5, shuffle=True,
                                       bert_model='bert-base-uncased',
                                       batch_size=32, seed=0):
    train_indexes, test_indexes = _cross_validation_indexes(
                                                df,
                                                n_splits=n_splits,
                                                shuffle=shuffle,
                                                random_state=seed)
    losses = []
    accuracies = []
    all_training_stats = []
    all_models = []
    for i in range(len(train_indexes)):
        print(f'Cross Validation: Fold {i+1}')
        print('------------------------------')
        training_stats, model = tmt.model_training_stratified_k_fold(
                                                      df,
                                                      train_indexes[i],
                                                      test_indexes[i],
                                                      device,
                                                      batch_size,
                                                      epochs,
        						learning_rate=2e-5,
        						epsilon=1e-8
        						)
        all_training_stats.append(training_stats)
        all_models.append(model)
    return all_training_stats, all_models


def _cross_validation_indexes(df, n_splits=5, shuffle=True, random_state=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    X = df['comment']
    y = df['label']
    train_indexes = []
    test_indexes = []

    for train_index, test_index in skf.split(X, y):
        train_indexes.append(train_index)
        test_indexes.append(test_index)
    return (train_indexes, test_indexes)


def get_stratified_tensors(df, train_indexes, test_indexes):
    sentences_train, labels_train = _get_dataset_subset(df, train_indexes)
    sentences_test, labels_test = _get_dataset_subset(df, test_indexes)

    input_ids_train, attention_masks_train, labels_train = bert_encoder.encode_dataset(
                                                        sentences_train,
                                                        labels_train,
                                                        max_phrase_len=250
                                                        )
    input_ids_test, attention_masks_test, labels_test = bert_encoder.encode_dataset(
                                                        sentences_test,
                                                        labels_test,
                                                        max_phrase_len=250
                                                        )
    training_data = {'input_ids': input_ids_train,
                     'attention_masks': attention_masks_train,
                     'labels': labels_train}
    test_data = {'input_ids': input_ids_test,
                 'attention_masks': attention_masks_test,
                 'labels': labels_test}
    return training_data, test_data


def _get_dataset_subset(df, subset_indexes):
    """Gets train or test sets from full dataset
    """
    tmp = df.loc[subset_indexes.tolist()]
    sentences = tmp['comment'].tolist()
    labels = tmp['label'].tolist()
    del tmp
    return sentences, labels


def cross_validation(tokenized_sentences, attention_masks, labels, cross_val_train_percentage):
    dataset = TensorDataset(tokenized_sentences, attention_masks, labels)
    train_size = int(cross_val_train_percentage * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    return train_dataset, val_dataset

