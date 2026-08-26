#!/usr/bin/env python3
"""
Bayesian Optimization module with optimize method
"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement

        Returns:
            X_next: numpy.ndarray of shape (1,) representing next sample point
            EI: numpy.ndarray of shape (ac_samples,) with expected improvement
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            y_opt = np.min(self.gp.Y)
            improvement = y_opt - mu - self.xsi
        else:
            y_opt = np.max(self.gp.Y)
            improvement = mu - y_opt - self.xsi

        with np.errstate(divide='ignore'):
            Z = np.where(sigma > 0, improvement / sigma, 0)
            EI = np.where(sigma > 0,
                          improvement * norm.cdf(Z) + sigma * norm.pdf(Z),
                          0)

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function using Bayesian Optimization

        Parameters:
            iterations: maximum number of iterations to perform

        Returns:
            X_opt: numpy.ndarray of shape (1,) representing optimal point
            Y_opt: numpy.ndarray of shape (1,) representing optimal value
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.isclose(self.gp.X, X_next)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            opt_idx = np.argmin(self.gp.Y)
        else:
            opt_idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[opt_idx]
        Y_opt = self.gp.Y[opt_idx]

        return X_opt, Y_opt
