#!/usr/bin/env python3
"""
Module for forward propagation with Dropout in a neural network.
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters:
    - X: numpy.ndarray of shape (nx, m) containing input data
    - weights: dictionary of weights and biases
    - L: number of layers in the network
    - keep_prob: probability that a node will be kept

    Returns:
    - cache: dictionary containing outputs of each layer and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Output layer uses softmax activation
            e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            cache['A' + str(i)] = e_Z / np.sum(e_Z, axis=0, keepdims=True)
        else:
            # Hidden layers use tanh activation and inverted dropout
            A = np.tanh(Z)
            D = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob).astype(
                int
            )
            A = (A * D) / keep_prob
            cache['D' + str(i)] = D
            cache['A' + str(i)] = A

    return cache
