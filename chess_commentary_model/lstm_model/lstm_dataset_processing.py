import pandas as pd
import numpy as np
import bert
import math

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

import sklearn
from sklearn.model_selection import StratifiedKFold

import lstm_model


def cross_validation_stratified_k_fold(df, epochs, vocabulary_size, embedding_output_dimensions,
                                       lstm_output_units_param, n_splits=5, shuffle=True,
                                       loss='binary_crossentropy', optimizer='adam',
                                       metrics=["accuracy"], batch_size=32, seed=0):
    """Executes a stratified k-fold cross-validation.
    Args:
    	df(DataFrame): dataset
    	epochs(int): number of epochs per training
    	vocabulary_size(int): vocabulary size
    	embedding_output_dimensions(int): number of output units in embedding layer of the model
	lstm_output_units_param(int): number of output units in lstm layer of the model
	n_splits(int): number of splits in stratified k-fold
	shuffle(bool): wether to shuffle or not shuffle the dataset before training
	loss(str): name of the used loss function
	optimizer(str): name of the used optimizer
	metrics(list): metrics to be analyzed by the model
	batch_size(int): batch size
	seed(int): seed number to be used in random operations.
    Returns:
    	losses(list): list of losses in each fold training
    	accuracies(list): list of accuracies in each fold training
    	cross_val_loss(float): mean loss of all folds training
    	cross_val_accuracy(float): mean accuracy of all folds training
    """
    train_indexes, test_indexes = cross_validation_indexes(
                            df, n_splits=n_splits, shuffle=shuffle, random_state=seed)
    losses = []
    accuracies = []
    for i in range(len(train_indexes)):
        print(f'Cross Validation: Fold {i}')
        print('------------------------------')
        train_data, test_data = get_batched_train_set(
                                        df,
                                        train_indexes[i],
                                        test_indexes[i],
                                        batch_size=batch_size,
                                        seed=seed)
        model = lstm_model.get_model(vocabulary_size, embedding_output_dimensions,
                          lstm_output_units_param, dropout_rate=0, dropout_seed=0)
        model = lstm_model.train_model(model, train_data, epochs=epochs, loss=loss,
                            optimizer=optimizer, metrics=metrics)

        y = lstm_model.evaluate_model(model, test_data, batch_size=batch_size)
        losses.append(y[0])
        accuracies.append(y[1])

        cross_val_loss = np.mean(losses)
        cross_val_accuracy = np.mean(accuracies)
    return losses, accuracies, cross_val_loss, cross_val_accuracy


def cross_validation_indexes(df, n_splits=5, shuffle=True, random_state=0):
    """Sorts dataset in stratified folds. Defines which rows will be in training or
    test sets by giving row indexes corresponding for each of these two sets.
    Args:
    	df(DataFrame): the dataset
    	n_splits(int): number of splits in cross validation
    	shuffle(bool): wether to shuffle the dataset or not
    	random_state(int): seed to be used if shuffle=True
    Returns:
        train_indexes(list): list of np.arrays, corresponding to each K-fold
        		      division for training set
        test_indexes(list): list of np.arrays, corresponding to each K-fold
                            division for test set
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    X = df['tokenized_commentary']
    y = df['label']
    train_indexes = []
    test_indexes = []

    for train_index, test_index in skf.split(X, y):
        train_indexes.append(train_index)
        test_indexes.append(test_index)
    return (train_indexes, test_indexes)


def get_batched_train_set(df, train_indexes, test_indexes, batch_size=32, seed=0):
    """Returns batched training and test sets iterators.
        Args:
            df(DataFrame): dataset
            train_indexes(list): indexes corresponding to trainining set rows
            test_indexes(list): indexes corresponding to test set rows
            batch_size(int): batch size
            seed(int): seed value
        Returns:
            train_data(obj): batched train set iterator
            test_data(obj): batched test set iterator
    """
    assert (not df['comment'].isna().any()), "Rows without comment are not allowed!"
    train_data = _batch_set(df, train_indexes, batch_size, seed=seed)
    test_data = _batch_set(df, test_indexes, batch_size, seed=seed)
    return train_data, test_data


def _batch_set(df, data_indexes, batch_size, seed=0):
    """Stores dataset in batches to be read while training the algorithm.
    Args:
        df(DataFrame): Dataframe
        data_indexes(list): indexes of training or test set
        batch_size(int): batch size value
        seed(int): seed number to be used in random operations
    """
    X = df.loc[data_indexes, 'tokenized_commentary'].tolist()
    Y = df.loc[data_indexes, 'label'].tolist()

    reviews = [[review, Y[i]] for i, review in enumerate(X)]
    np.random.seed(seed); np.random.shuffle(reviews)

    reviews_labels = [(review_lab[0], review_lab[1]) for review_lab in reviews]
    processed_dataset = tf.data.Dataset.from_generator(
            lambda: reviews_labels, output_types=(tf.int32, tf.int32))

    batched_dataset = processed_dataset.padded_batch(batch_size, padded_shapes=((None, ), ()))
    total_batches = math.ceil(len(reviews) / batch_size)
    batched_dataset.shuffle(total_batches)
    batched_data = batched_dataset.take(total_batches)
    return batched_data

