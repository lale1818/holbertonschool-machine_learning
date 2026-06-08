#!/usr/bin/env python3
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Class constructor"""
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.X = data
        n = data.shape[1]
        self.cov = np.dot((data - self.mean), (data - self.mean).T) / (n - 1)
        self.d = data.shape[0]

    def pdf(self, x):
        """Calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (self.d, 1):
            raise ValueError(f"x must have the shape ({self.d}, 1)")

        d = self.d
        mean = self.mean
        cov = self.cov

        diff = x - mean
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)

        exponent = -0.5 * np.matmul(np.matmul(diff.T, inv), diff)
        exponent = exponent.item()

        denom = np.sqrt(((2 * np.pi) ** d) * det)

        return np.exp(exponent) / denom
