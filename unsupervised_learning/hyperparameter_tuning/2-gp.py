#!/usr/bin/env python3
"""
Gaussian Process module with update
"""
import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Constructor for GaussianProcess

        Parameters:
            X_init: numpy.ndarray of shape (t, 1) containing inputs sampled
            Y_init: numpy.ndarray of shape (t, 1) containing outputs
            l: length parameter for the kernel
            sigma_f: standard deviation given to output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF) kernel

        Parameters:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            Covariance kernel matrix of shape (m, n)
        """
        sq_sum1 = np.sum(X1 ** 2, 1).reshape(-1, 1)
        sq_sum2 = np.sum(X2 ** 2, 1)
        dot_prod = 2 * np.dot(X1, X2.T)
        sqdist = sq_sum1 + sq_sum2 - dot_prod
        return self.sigma_f ** 2 * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a Gaussian process

        Parameters:
            X_s: numpy.ndarray of shape (s, 1) containing sample points

        Returns:
            mu: numpy.ndarray of shape (s,) containing the mean
            sigma: numpy.ndarray of shape (s,) containing the variance
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process with a new point

        Parameters:
            X_new: numpy.ndarray of shape (1,) representing new sample point
            Y_new: numpy.ndarray of shape (1,) representing new output value
        """
        X_new_reshaped = X_new.reshape(-1, 1)
        Y_new_reshaped = Y_new.reshape(-1, 1)

        self.X = np.vstack((self.X, X_new_reshaped))
        self.Y = np.vstack((self.Y, Y_new_reshaped))
        self.K = self.kernel(self.X, self.X)
