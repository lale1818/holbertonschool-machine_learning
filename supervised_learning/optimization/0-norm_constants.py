#!/usr/bin/env python3
"""
This module contains a function that calculates the normalization constants
of a matrix.
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Parameters:
    - X: numpy.ndarray of shape (m, nx) to normalize

    Returns:
    - mean: numpy.ndarray of shape (nx,) with the mean of each feature
    - std: numpy.ndarray of shape (nx,) with standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
