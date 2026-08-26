#!/usr/bin/env python3
"""
Model training module with ModelCheckpoint (save best) using Keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional
    Early Stopping, Learning Rate Decay, and Model Checkpointing

    Parameters:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: tuple of (X_valid, Y_valid) to validate model with
        early_stopping: boolean indicating whether early stopping should be used
        patience: patience used for early stopping
        learning_rate_decay: boolean indicating whether LR decay should be used
        alpha: initial learning rate
        decay_rate: decay rate
        save_best: boolean indicating whether to save the best model iteration
        filepath: path where the model should be saved
        verbose: boolean that determines if output should be printed
        shuffle: boolean that determines whether to shuffle batches each epoch

    Returns:
        the History object generated after training the model
    """
    callbacks = []

    if validation_data is not None:
        if early_stopping:
            early_stop_callback = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
            callbacks.append(early_stop_callback)

        if learning_rate_decay:
            def scheduler(epoch):
                """Inverse time decay function"""
                return alpha / (1 + decay_rate * epoch)

            lr_scheduler_callback = K.callbacks.LearningRateScheduler(
                scheduler,
                verbose=1
            )
            callbacks.append(lr_scheduler_callback)

        if save_best and filepath is not None:
            checkpoint_callback = K.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True
            )
            callbacks.append(checkpoint_callback)

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
