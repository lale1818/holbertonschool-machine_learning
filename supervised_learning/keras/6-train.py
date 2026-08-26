#!/usr/bin/env python3
"""
Model training module with Early Stopping using Keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and optionally
    uses Early Stopping based on validation loss

    Parameters:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: tuple of (X_valid, Y_valid) to validate model with
        early_stopping: boolean indicating whether early stopping should be used
        patience: patience used for early stopping
        verbose: boolean that determines if output should be printed
        shuffle: boolean that determines whether to shuffle batches each epoch

    Returns:
        the History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop_callback)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
