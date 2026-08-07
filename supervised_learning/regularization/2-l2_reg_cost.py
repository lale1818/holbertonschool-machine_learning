#!/usr/bin/env python3
"""
Module to calculate the total cost of a neural network with L2 regularization
for each layer in Keras.
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    - cost: tensor containing the cost of the network without L2 regularization
    - model: Keras model that includes layers with L2 regularization

    Returns:
    - a tensor containing the total cost for each layer of the network,
      accounting for L2 regularization
    """
    reg_losses = []
    for layer in model.layers:
        for loss in layer.losses:
            reg_losses.append(cost + loss)

    return tf.reshape(tf.stack(reg_losses), [-1])
