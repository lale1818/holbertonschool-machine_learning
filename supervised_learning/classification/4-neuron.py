#!/usr/bin/env python3
"""
This module defines a single neuron performing
binary classification with evaluation.
"""
import numpy as np


class Neuron:
    """
    Represents a single neuron with private weights,
    bias, and activation output.
    """

    def __init__(self, nx):
        """
        Initializes the neuron with input features.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter for private weights vector __W """
        return self.__W

    @property
    def b(self):
        """ Getter for private bias __b """
        return self.__b

    @property
    def A(self):
        """ Getter for private activated output __A """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates forward propagation using sigmoid.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates cost using logistic regression.
        """
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.sum(loss) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions and cost.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
