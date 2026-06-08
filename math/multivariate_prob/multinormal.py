#!/usr/bin/env python3
"""Defines the class MultiNormal that represents a Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Initializes the distribution variables with data shape (d, n)"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Orta qiyməti hesablayırıq, forması (d, 1) olmalıdır
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Datanı mərkəzləşdiririk
        data_centered = data - self.mean

        # Kovariasiya: (data_centered @ data_centered.T) / (n - 1)
        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a given data point x"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        # Düsturun əmsal hissəsi: 1 / sqrt((2*pi)^d * det)
        denominator = np.sqrt(((2 * np.pi) ** d) * det)

        # Eksponent daxilindəki hissə: -0.5 * (x - mean).T @ cov^-1 @ (x - mean)
        x_centered = x - self.mean
        exponent = -0.5 * np.matmul(np.matmul(x_centered.T, inv), x_centered)

        # PDF dəyərini tək bir float element olaraq çıxarırıq [0][0]
        pdf_value = np.exp(exponent) / denominator

        return pdf_value[0][0]
