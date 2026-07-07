#!/usr/bin/env python3
"""
This module defines a single neuron performing
binary classification with private attributes.
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
