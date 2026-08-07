#!/usr/bin/env python3
"""
This module contains a function that sets up the gradient descent
with momentum optimization algorithm in TensorFlow.
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm.

    Parameters:
    - alpha: learning rate
    - beta1: momentum weight

    Returns:
    - optimizer: the SGD optimizer instance with momentum
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
