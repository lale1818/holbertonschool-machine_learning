#!/usr/bin/env python3
"""
This module contains a function that sets up the RMSProp optimization
algorithm in TensorFlow.
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.

    Parameters:
    - alpha: learning rate
    - beta2: RMSProp weight (discounting factor)
    - epsilon: small number to avoid division by zero

    Returns:
    - optimizer: RMSProp optimizer instance
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
