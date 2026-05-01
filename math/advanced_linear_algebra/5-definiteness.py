#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    
    # Definiteness is typically defined for symmetric matrices
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigenvalues = np.linalg.eigvals(matrix)
        
        if np.all(eigenvalues > 0):
            return "Positive definite"
        if np.all(eigenvalues < 0):
            return "Negative definite"
        if np.all(eigenvalues >= 0):
            return "Positive semi-definite"
        if np.all(eigenvalues <= 0):
            return "Negative semi-definite"
        
        pos = any(eigenvalues > 0)
        neg = any(eigenvalues < 0)
        if pos and neg:
            return "Indefinite"
    except Exception:
        return None

    return None
