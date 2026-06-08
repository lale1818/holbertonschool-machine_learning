#!/usr/bin/env python3
"""Calculates a correlation matrix"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Diaqonal elementlərin kökünü alıb standart meylləri tapırıq (std_devs)
    d = np.sqrt(np.diag(C))

    # Xətti asılılığı hesablamaq üçün xarici hasil (outer product) edirik: d_i * d_j
    outer_v = np.outer(d, d)

    # Korrelyasiya = C / (d_i * d_j)
    correlation_matrix = C / outer_v

    return correlation_matrix
