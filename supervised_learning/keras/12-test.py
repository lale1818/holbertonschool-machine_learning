#!/usr/bin/env python3
"""
Module to test a neural network model using Keras.
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network model.

    Args:
        network: the network model to test
        data: the input data to test the model with
        labels: correct one-hot labels of data
        verbose: boolean determining if output should be printed

    Returns:
        the loss and accuracy of the model with the testing data, respectively
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
