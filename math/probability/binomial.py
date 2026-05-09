#!/usr/bin/env python3
"""Contains the Binomial class to represent a binomial distribution"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes the binomial distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Mean (mu) hesablamaq
            mean = sum(data) / len(data)

            # Variance (sigma^2) hesablamaq
            diff_sum = sum([(x - mean) ** 2 for x in data])
            variance = diff_sum / len(data)

            # Parametrləri təxmin etmək
            p_val = 1 - (variance / mean)
            self.n = int(round(mean / p_val))
            self.p = float(mean / self.n)
