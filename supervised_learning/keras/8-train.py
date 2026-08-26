#!/usr/bin/env python3
"""
Trains a Keras model and saves the best model based on validation loss.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with early stopping,
    learning rate decay, and saving the best iteration model.
    """
    callbacks = []

    if validation_data:
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            )
            callbacks.append(early_stop)

        if learning_rate_decay:
            def scheduler(epoch):
                """Calculates inverse time decay learning rate per epoch."""
                return alpha / (1 + decay_rate * epoch)

            lr_decay = K.callbacks.LearningRateScheduler(
                scheduler,
                verbose=1
            )
            callbacks.append(lr_decay)

        if save_best and filepath:
            m_checkpoint = K.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True
            )
            callbacks.append(m_checkpoint)

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
