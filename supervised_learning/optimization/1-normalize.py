#!/usr/bin/env python3
"""
This module contains a function that normalizes a matrix.
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters:
    - X: numpy.ndarray of shape (d, nx) to normalize
    - m: numpy.ndarray of shape (nx,) containing feature means
    - s: numpy.ndarray of shape (nx,) containing feature standard deviations

    Returns:
    - The normalized X matrix
    """
    return (X - m) / s
