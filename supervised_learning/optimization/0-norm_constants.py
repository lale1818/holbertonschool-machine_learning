#!/usr/bin/env python3
"""
Module containing the normalization_constants function.
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Parameters:
        X (numpy.ndarray): Matrix of shape (m, nx) to normalize

    Returns:
        tuple: (mean, std) of each feature across columns
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
