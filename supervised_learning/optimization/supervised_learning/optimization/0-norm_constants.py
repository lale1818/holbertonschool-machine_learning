#!/usr/bin/env python3
"""
Calculates the normalization constants of a matrix
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation of each feature in X
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
