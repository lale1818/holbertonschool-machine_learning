#!/usr/bin/env python3
"""
Optimization module for Keras models
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model with
    categorical crossentropy loss and accuracy metrics

    Parameters:
        network: the model to optimize
        alpha: learning rate
        beta1: first Adam optimization parameter
        beta2: second Adam optimization parameter

    Returns:
        None
    """
    opt = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
