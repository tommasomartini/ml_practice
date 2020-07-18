import numpy as np


class Linear:

    @staticmethod
    def fw(x):
        return x

    @staticmethod
    def grad(x):
        return np.ones(x.shape)


class ReLU:

    @staticmethod
    def fw(x):
        return np.clip(x, 0, None)

    @staticmethod
    def grad(x):
        return (x >= 0).astype(float)


class LeakyReLU:
    p = 0.01

    @staticmethod
    def fw(x):
        return np.zeros(x.shape) + x * (x >= 0) + LeakyReLU.p * x * (x < 0)

    @staticmethod
    def grad(x):
        return np.zeros(x.shape) + 1 * (x >= 0) + LeakyReLU.p * (x < 0)


class Tanh:

    @staticmethod
    def fw(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def grad(x):
        return 1 - Tanh.fw(x) ** 2


class Logistic:

    @staticmethod
    def fw(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def grad(x):
        return Logistic.fw(x) * (1 - Logistic.fw(x))


Sigmoid = Logistic
