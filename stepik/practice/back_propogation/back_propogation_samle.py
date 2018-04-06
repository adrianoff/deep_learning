# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))

def get_error(deltas, sums, weights):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)

    https://habrahabr.ru/post/198268/
    """

    # ValueError: ('array([[ 0.3,  0.2],\n       [ 0.3,  0.2]])', 'array([[0, 1, 1],\n       [0, 2, 2]])', 'array([[ 0.7,  0.2,  0.7],\n       [ 0.8,  0.3,  0.6]])')

    delta = deltas.dot(weights)
    prime = sigmoid_prime(sums)

    return np.mean(delta * prime, 0)


deltas = np.array([[0.3,  0.2], [0.3,  0.2]])
sums = np.array([[0, 1, 1], [0, 2, 2]])
weights = np.array([[0.7,  0.2,  0.7], [0.8,  0.3,  0.6]])

rez = get_error(deltas, sums, weights)
pass