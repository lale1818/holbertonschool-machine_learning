#!/usr/bin/env python3
"""
Module to create a layer in TensorFlow with Dropout regularization.
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Parameters:
    - prev: tensor containing the output of the previous layer
    - n: number of nodes the new layer should contain
    - activation: activation function for the new layer
    - keep_prob: probability that a node will be kept
    - training: boolean indicating whether the model is in training mode

    Returns:
    - the output of the new layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )

    A = dense_layer(prev)

    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)

    return dropout_layer(A, training=training)
