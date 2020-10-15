##############################
## BERT - Model
##############################
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


def cross_validation(input_ids, attention_masks, labels, cross_val_train_percentage):
  dataset = TensorDataset(input_ids, attention_masks, labels)
  train_size = int(cross_val_train_percentage * len(dataset))
  val_size = len(dataset) - train_size

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  print('{:>5,} training samples'.format(train_size))
  print('{:>5,} validation samples'.format(val_size))

  return train_dataset, val_dataset


def data_loader(batch_size, train_dataset, val_dataset):
    train_dataloader = DataLoader(
              train_dataset,
              sampler = RandomSampler(train_dataset),
              batch_size = batch_size
          )

    validation_dataloader = DataLoader(
              val_dataset,
              sampler = SequentialSampler(val_dataset),
              batch_size = batch_size
          )
    return train_dataloader, validation_dataloader


def get_model(bert_model='bert-base-uncased', num_labels=2):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=bert_model,
        num_labels=num_labels,
        output_attentions = False,
        output_hidden_states = False
    )
    model.cuda()
    return model


def get_adam_optimizer(model, learning_rate=2e-5, epsilon=1e-8):
  optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )
  return optimizer


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_epoch(train_dataloader, optimizer, scheduler, model, total_train_loss, device):
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()
        loss, logits = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        del b_input_ids
        del b_input_mask
        del b_labels
    return (optimizer, scheduler, total_train_loss)


def validate_epoch(validation_dataloader, model, device):
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
            (loss, logits) = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        del b_input_ids
        del b_input_mask
        del b_labels
    return total_eval_accuracy, total_eval_loss


def classification_iterate(train_dataloader, validation_dataloader, optimizer,
                           scheduler, model, device, training_stats, epochs):
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        total_train_loss = 0
        model.train()

        optimizer, scheduler, total_train_loss = train_epoch(train_dataloader,
                                                             optimizer,
                                                             scheduler,
                                                             model,
                                                             total_train_loss,
                                                             device
                                                             )
        print_gpu_space()
        avg_train_loss = total_train_loss / len(train_dataloader)

        print(" Average training loss: {0:.2f}".format(avg_train_loss))

        print("")
        print("Running Validation...")
        model.eval()

        total_eval_accuracy, total_eval_loss = validate_epoch(
                                          validation_dataloader, model, device)
        print_gpu_space()
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

        print(" Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        print(" Validation Loss: {0:.2f}".format(avg_val_loss))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy
            }
        )

    print("")
    print("Training complete!")

    return (training_stats, model)


def model_training(input_ids, attention_masks, labels, device, batch_size, epochs,
                   learning_rate=2e-5, epsilon=1e-8, cross_val_train_percentage=0.7):
    model = get_model(bert_model='bert-base-uncased', num_labels=2)
    train_dataset, val_dataset = cross_validation(input_ids, attention_masks, labels,
                                                  cross_val_train_percentage)
    train_dataloader, validation_dataloader = data_loader(batch_size=batch_size,
                                    train_dataset=train_dataset, val_dataset=val_dataset)
    optimizer = get_adam_optimizer(model, learning_rate=learning_rate, epsilon=epsilon)
    

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    training_stats = []
    training_stats, model = classification_iterate(train_dataloader,
                                 validation_dataloader, optimizer, scheduler,
                                 model, device, training_stats, epochs)
    return (training_stats, model)


def load_test_set(df):
    sentences = df['comment'].to_list()
    labels = df['label'].to_list()

    input_ids = []
    attention_masks = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True,
                            max_length = 250,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation=True
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    batch_size = 32  

    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler,
                                       batch_size=batch_size)
    return prediction_dataloader


def run_model(model, prediction_dataloader, device):
    model.eval()
    predictions , true_labels = [], []

    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)
    return flat_predictions, flat_true_labels


def print_gpu_space():
    GPUs = GPU.getGPUs()
    # XXX: only one GPU on Colab and isn’t guaranteed
    gpu = GPUs[0]
    def printm():
        process = psutil.Process(os.getpid())
        print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    printm()
