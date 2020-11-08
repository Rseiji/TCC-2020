import pandas as pd
import numpy as np
import bert
import math

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers


def get_model(vocabulary_size, embedding_output_dimensions, lstm_output_units_param,
              dropout_rate=0, dropout_seed=0):
    """Returns bi-lstm model
    	Args:
    	    vocabulary_size(int): size of tokenizer vocabulary
    	    embedding_output_dimensions(int): embedding output dimension
    	    lstm_output_units_param(int): lstm output dimension
    	    dropout_rate(real): value between 0.1, defining the dropout layer rate
    	    dropout_seed(int): dropout seed
    """
    return tf.keras.Sequential([
            layers.Embedding(vocabulary_size, embedding_output_dimensions),
            layers.Bidirectional(layers.LSTM(lstm_output_units_param)),
            layers.Dropout(rate=dropout_rate, seed=dropout_seed),
            layers.Dense(1, activation='sigmoid')
        ])


def train_model(model, train_data, epochs, loss='binary_crossentropy',
                optimizer='adam', metrics=["accuracy"]):
    """executes model training. Compiles the model and fits it to training
    dataset.
        Args:
            model(obj): bi-lstm model
            train_data: training dataset. An iterator used to batch training
            epochs(int): number of training epochs
            loss(str): loss function. Default binary crossentropy
            optimizer(str): algorithm optimizer. Default adam
            metrics(list): list of model evaluation metrics
        Returns:
            model(boj): The trained model
    """
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_data, epochs=epochs)
    return model


def evaluate_model(model, test_data, batch_size):
    return model.evaluate(test_data, batch_size=batch_size)


def get_model_prediction(model, test_data):
    return model.predict(test_data)

