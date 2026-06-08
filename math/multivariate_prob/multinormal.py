#!/usr/bin/env python3
"""Defines Multivariate Normal distribution class"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Initializes with data of shape (d, n)"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Mean calculation with shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center data points
        data_centered = data - self.mean

        # Covariance calculation
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """Calculates PDF at a data point x"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # Formula parts split to prevent line too long (E501)
        denom = np.sqrt(((2 * np.pi) ** d) * det)
        x_centered = x - self.mean
        exponent = -0.5 * np.matmul(np.matmul(x_centered.T, inv), x_centered)

        pdf_value = np.exp(exponent) / denom

        return pdf_value[0][0]
