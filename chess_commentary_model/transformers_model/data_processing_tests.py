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

    return df_training, df_test


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

    return df_training, df_test


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

    return df_training, df_test
