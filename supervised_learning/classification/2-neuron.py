#!/usr/bin/env python3
"""
This module defines a single neuron performing binary classification
with forward propagation.
"""
import numpy as np


class Neuron:
    """
    Represents a single neuron with private weights, bias, and activation output
    """

    def __init__(self, nx):
        """
        Initializes the neuron with the given number of input features.
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
        """ Getter for the private weights vector __W """
        return self.__W

    @property
    def b(self):
        """ Getter for the private bias __b """
        return self.__b

    @property
    def A(self):
        """ Getter for the private activated output __A """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron using sigmoid function.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
