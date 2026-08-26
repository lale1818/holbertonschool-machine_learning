#!/usr/bin/env python3
"""
Model training module with validation data using Keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and analyzes
    validation data

    Parameters:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: tuple of (X_valid, Y_valid) to validate model with
        verbose: boolean that determines if output should be printed
        shuffle: boolean that determines whether to shuffle batches each epoch

    Returns:
        the History object generated after training the model
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
