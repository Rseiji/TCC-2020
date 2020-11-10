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

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import transformers_model_utils as tm
import transformers_model_execution as tme
import transformers_cross_validation as tcv


def model_training_normal_cross_validation(tokenized_sentences, attention_masks,
                                           labels, device, batch_size, epochs,
                                           learning_rate=2e-5, epsilon=1e-8,
                                           cross_val_train_percentage=0.7):
    model = tm.get_model(bert_model='bert-base-uncased', num_labels=2)
    optimizer = tm.get_adam_optimizer(model, learning_rate=learning_rate, epsilon=epsilon)

    train_dataset, val_dataset = tcv.cross_validation(tokenized_sentences, attention_masks,
                                                  labels, cross_val_train_percentage
                                                  )
    train_dataloader, validation_dataloader = tm.get_data_loader(
                                                  batch_size=batch_size,
                                                  train_dataset=train_dataset,
                                                  val_dataset=val_dataset
                                                  )

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    training_stats = []
    training_stats, model = tme.classification_iterate(train_dataloader,
                                 validation_dataloader, optimizer, scheduler,
                                 model, device, training_stats, epochs)
    return (training_stats, model)


def model_training_stratified_k_fold(df, train_indexes, test_indexes, device,
                                     batch_size, epochs, learning_rate=2e-5,
                                     epsilon=1e-8, bert_model='bert-base-uncased'):
    model = tm.get_model(bert_model=bert_model, num_labels=2)
    optimizer = tm.get_adam_optimizer(model, learning_rate=learning_rate, epsilon=epsilon)

    train_data, test_data = tcv.get_stratified_tensors(df, train_indexes,
                                                       test_indexes)

    train_dataset, _ = tcv.cross_validation(train_data['input_ids'],
                                            train_data['attention_masks'],
                                            train_data['labels'], 1)
    test_dataset, _ = tcv.cross_validation(test_data['input_ids'],
                                           test_data['attention_masks'],
                                           test_data['labels'], 1)

    train_dataloader, test_dataloader = tm.get_data_loader(
                                            batch_size=batch_size,
                                            train_dataset=train_dataset,
                                            val_dataset=test_dataset
                                            )

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    training_stats = []
    training_stats, model = tme.classification_iterate(train_dataloader,
                                test_dataloader, optimizer, scheduler,
                                model, device, training_stats, epochs)
    return (training_stats, model)

