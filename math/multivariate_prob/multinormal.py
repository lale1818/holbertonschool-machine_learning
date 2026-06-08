#!/usr/bin/env python3
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Class constructor"""
        self.X = data
        self.mean = np.mean(data, axis=1, keepdims=True)
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

        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)

        exponent = -0.5 * (diff.T @ cov_inv @ diff)[0, 0]

        coeff = 1 / np.sqrt(((2 * np.pi) ** d) * cov_det)

        return coeff * np.exp(exponent)
