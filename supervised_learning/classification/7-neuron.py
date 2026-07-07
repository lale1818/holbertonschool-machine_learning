#!/usr/bin/env python3
"""
This module defines a single neuron performing
binary classification with upgraded training capabilities.
"""
import matplotlib.pyplot as plt
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neuron with verbose logging and graphing.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        for i in range(iterations):
            A = self.forward_prop(X)
            
            if i % step == 0:
                cost = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    costs.append(cost)
                    iters.append(i)
            
            self.gradient_descent(X, Y, A, alpha)

        A_final = self.forward_prop(X)
        cost_final = self.cost(Y, A_final)

        if verbose:
            print("Cost after {} iterations: {}".format(iterations,
                                                       cost_final))
        if graph:
            costs.append(cost_final)
            iters.append(iterations)
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
