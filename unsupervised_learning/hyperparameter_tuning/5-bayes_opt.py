#!/usr/bin/env python3
"""Defines the BayesianOptimization class for hyperparameter tuning"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, acq_samples,
                 l=1, sigma_f=1, xsi=0.01, mode='MIN'):
        """Initializes the Bayesian Optimization"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               num=acq_samples).reshape(-1, 1)
        self.xsi = xsi
        self.mode = mode

    def acquisition(self):
        """Calculates the next best sample point using Expected Improvement"""
        from scipy.stats import norm

        mu, sigma = self.gp.predict(self.X_s)

        if self.mode == 'MIN':
            Y_sample_opt = np.min(self.gp.Y)
            imp = Y_sample_opt - mu - self.xsi
        else:
            Y_sample_opt = np.max(self.gp.Y)
            imp = mu - Y_sample_opt - self.xsi

        Z = np.zeros_like(sigma)
        pos_mask = sigma > 0
        Z[pos_mask] = imp[pos_mask] / sigma[pos_mask]

        EI = np.zeros_like(sigma)
        EI[pos_mask] = (imp[pos_mask] * norm.cdf(Z[pos_mask]) +
                        sigma[pos_mask] * norm.pdf(Z[pos_mask]))

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.

        Args:
            iterations (int): Maximum number of iterations to perform.

        Returns:
            X_opt (numpy.ndarray): Optimal point of shape (1,)
            Y_opt (numpy.ndarray): Optimal function value of shape (1,)
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Yeni nöqtə artıq yoxlanılıbsa erkən dayandırırıq
            if np.any(np.isclose(X_next, self.gp.X)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.mode == 'MIN':
            opt_idx = np.argmin(self.gp.Y)
        else:
            opt_idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[opt_idx]
        Y_opt = self.gp.Y[opt_idx]

        return X_opt, Y_opt
