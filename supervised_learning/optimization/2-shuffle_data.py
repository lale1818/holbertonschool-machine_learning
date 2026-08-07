#!/usr/bin/env python3
"""
This module contains a function that shuffles the data points
in two matrices in the same way.
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters:
    - X: numpy.ndarray of shape (m, nx)
    - Y: numpy.ndarray of shape (m, ny)

    Returns:
    - The shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
