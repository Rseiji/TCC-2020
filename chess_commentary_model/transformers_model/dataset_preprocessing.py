"""Métodos de preprocessamento de testes individuais
"""

import pandas as pd
import numpy as np
import math


def test_1(df, seed=0):
    """training: balanced; test: balanced
        training: 80k (40k 0, 40k 1)
        test: 20k (10k 0, 10k 1)
    """
    df_ones = df[df['label'] == 1]
    df_zeros = df[df['label'] == 0]

    df_ones = df_ones.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_zeros = df_zeros.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_ones_training = df_ones.loc[:40000]
    df_zeros_training = df_zeros.loc[:40000]
    df_ones_test = df_ones.loc[40000:50000]
    df_zeros_test = df_zeros.loc[40000:50000]

    df_training = pd.concat([df_ones_training, df_zeros_training])
    df_training = df_training.sample(frac=1).reset_index(drop=True)
    df_test = pd.concat([df_ones_test, df_zeros_test])
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    sentences_train = df_training['comment'].tolist()
    sentences_test = df_test['comment'].tolist() 
    labels_train = df_training['label'].tolist()
    labels_test = df_test['label'].tolist()
    return sentences_train, sentences_test, labels_train, labels_test


def test_2(df, seed=0):
    """training: balanced; test: unbalanced
       training: 80k (40k 0, 40k 1)
       test: 20k (4k 0, 16k 1)
    """
    df_ones = df[df['label'] == 1]
    df_zeros = df[df['label'] == 0]

    df_ones = df_ones.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_zeros = df_zeros.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_ones_training = df_ones.loc[:40000]
    df_zeros_training = df_zeros.loc[:40000]
    df_ones_test = df_ones.loc[40000:44000]
    df_zeros_test = df_zeros.loc[40000:56000]

    df_training = pd.concat([df_ones_training, df_zeros_training])
    df_training = df_training.sample(frac=1).reset_index(drop=True)
    df_test = pd.concat([df_ones_test, df_zeros_test])
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    sentences_train = df_training['comment'].tolist()
    sentences_test = df_test['comment'].tolist() 
    labels_train = df_training['label'].tolist()
    labels_test = df_test['label'].tolist()
    return sentences_train, sentences_test, labels_train, labels_test


def test_3(df, seed=0):
    """training: unbalanced; test: unbalanced
        training: 80k (16k 1, 64k 0)
        test: 20k (4k 1, 16k 0)
    """
    df_ones = df[df['label'] == 1]
    df_zeros = df[df['label'] == 0]

    df_ones = df_ones.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_zeros = df_zeros.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_ones_training = df_ones.loc[:16000]
    df_zeros_training = df_zeros.loc[:64000]
    df_ones_test = df_ones.loc[16000:20000]
    df_zeros_test = df_zeros.loc[64000:80000]

    df_training = pd.concat([df_ones_training, df_zeros_training])
    df_training = df_training.sample(frac=1).reset_index(drop=True)
    df_test = pd.concat([df_ones_test, df_zeros_test])
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    sentences_train = df_training['comment'].tolist()
    sentences_test = df_test['comment'].tolist()
    labels_train = df_training['label'].tolist()

    labels_test = df_test['label'].tolist()
    return sentences_train, sentences_test, labels_train, labels_test


##################################
## Tests on old dataset
##################################

def test_4(df, seed=0):
    """ training: balanced; test: balanced
        training: 58k (29k 0, 29k 1)
        test: 14.5k (7.25k 0, 7.25k 1)
    """
    df_ones = df[df['label'] == 1]
    df_zeros = df[df['label'] == 0]

    df_ones = df_ones.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_zeros = df_zeros.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_ones_training = df_ones.loc[:29000]
    df_zeros_training = df_zeros.loc[:29000]
    df_ones_test = df_ones.loc[29000:36250]
    df_zeros_test = df_zeros.loc[29000:36250]

    df_training = pd.concat([df_ones_training, df_zeros_training])
    df_training = df_training.sample(frac=1).reset_index(drop=True)
    df_test = pd.concat([df_ones_test, df_zeros_test])
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    sentences_train = df_training['comment'].tolist()
    sentences_test = df_test['comment'].tolist()
    labels_train = df_training['label'].tolist()

    labels_test = df_test['label'].tolist()
    return sentences_train, sentences_test, labels_train, labels_test


def test_5(df, seed=0):
    """training: balanced; test: unbalanced
       training: 58k (29000 0, 29000 1)
       test: 14.5k (12905 0, 1595 1)
    """
    df_ones = df[df['label'] == 1]
    df_zeros = df[df['label'] == 0]

    df_ones = df_ones.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_zeros = df_zeros.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_ones_training = df_ones.loc[:29000]
    df_zeros_training = df_zeros.loc[:29000]
    df_ones_test = df_ones.loc[29000:30595]
    df_zeros_test = df_zeros.loc[29000:41905]

    df_training = pd.concat([df_ones_training, df_zeros_training])
    df_training = df_training.sample(frac=1).reset_index(drop=True)
    df_test = pd.concat([df_ones_test, df_zeros_test])
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    sentences_train = df_training['comment'].tolist()
    sentences_test = df_test['comment'].tolist() 
    labels_train = df_training['label'].tolist()
    labels_test = df_test['label'].tolist()
    return sentences_train, sentences_test, labels_train, labels_test


def test_6(df, seed=0):
    """training: unbalanced; test: unbalanced
        training: 58k (6380 1, 51620 0)
        test: 14.5k (1595 1, 12905 0)
    """
    df_ones = df[df['label'] == 1]
    df_zeros = df[df['label'] == 0]

    df_ones = df_ones.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_zeros = df_zeros.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_ones_training = df_ones.loc[:6380]
    df_zeros_training = df_zeros.loc[:51620]
    df_ones_test = df_ones.loc[6380:7975]
    df_zeros_test = df_zeros.loc[51620:64525]

    df_training = pd.concat([df_ones_training, df_zeros_training])
    df_training = df_training.sample(frac=1).reset_index(drop=True)
    df_test = pd.concat([df_ones_test, df_zeros_test])
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    sentences_train = df_training['comment'].tolist()
    sentences_test = df_test['comment'].tolist()
    labels_train = df_training['label'].tolist()

    labels_test = df_test['label'].tolist()
    return sentences_train, sentences_test, labels_train, labels_test

