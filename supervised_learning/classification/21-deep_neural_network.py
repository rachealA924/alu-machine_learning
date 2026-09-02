#!/usr/bin/env python3
"""Defines a DeepNeuralNetwork class"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor

        Args:
            nx (int): number of input features
            layers (list): number of nodes in each layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights['W' + str(i + 1)] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx))
            else:
                self.__weights['W' + str(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) *
                    np.sqrt(2 / layers[i - 1]))

            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            W = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]
            A_prev = self.__cache['A' + str(i)]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(i + 1)] = A

        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) with correct labels
            A: numpy.ndarray with shape (1, m) with activated output
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
            Y: numpy.ndarray with shape (1, m) with correct labels
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Args:
            Y: numpy.ndarray with shape (1, m) with correct labels
            cache: dictionary containing intermediary values of the network
            alpha: learning rate
        """
        m = Y.shape[1]
        L = self.__L
        weights = self.__weights.copy()
        dZ = cache['A' + str(L)] - Y

        for i in range(L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            W = weights['W' + str(i)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if i > 1:
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))

            self.__weights['W' + str(i)] = (
                weights['W' + str(i)] - alpha * dW)
            self.__weights['b' + str(i)] = (
                weights['b' + str(i)] - alpha * db)
