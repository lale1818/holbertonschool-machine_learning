#!/usr/bin/env python3
"""
Bayesian Optimization module
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Constructor for BayesianOptimization

        Parameters:
            f: black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1) containing initial inputs
            Y_init: numpy.ndarray of shape (t, 1) containing initial outputs
            bounds: tuple of (min, max) representing space bounds
            ac_samples: number of samples to analyze during acquisition
            l: length parameter for the kernel
            sigma_f: standard deviation given to output
            xsi: exploration-exploitation factor for acquisition
            minimize: bool for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
