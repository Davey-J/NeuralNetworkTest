import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt


def leaky_relu(x, leak_factor):
    return np.where(x > 0, x, x * leak_factor)


def d_leaky_relu(x, leak_factor):
    return np.where(x > 0, 1.0, leak_factor)


def relu(x):
    return x * (x > 0)


def d_relu(x):
    return 1. * (x > 0)


def sigmoid(x):
    return 1/(1+(np.exp((-x))))


def d_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def d_softmax(x):
    raise NotImplementedError