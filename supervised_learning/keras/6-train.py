#!/usr/bin/env python3
"""
Trains a Keras model using early stopping.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with early stopping.

    Parameters:
    - network: the model to train
    - data: numpy.ndarray of shape (m, nx) containing input data
    - labels: one-hot numpy.ndarray of shape (m, classes) containing labels
    - batch_size: size of the batch used for mini-batch gradient descent
    - epochs: number of passes through data
    - validation_data: tuple (X_valid, Y_valid) for validation data
    - early_stopping: boolean indicating whether to use early stopping
    - patience: patience count used for early stopping
    - verbose: boolean determining whether to output during training
    - shuffle: boolean determining whether to shuffle batches every epoch

    Returns:
    - History object generated after training the model
    """
    callbacks = []

    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

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
