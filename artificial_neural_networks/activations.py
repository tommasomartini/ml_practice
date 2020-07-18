import numpy as np


class ReLU:

    @staticmethod
    def fw(x):
        return np.clip(x, 0, None)

    @staticmethod
    def grad(x):
        return (x >= 0).astype(float)


class Logistic:

    @staticmethod
    def fw(x):
        return np.exp(x) / (1 + np.exp(x))

    @staticmethod
    def grad(x):
        return Logistic.fw(x) * Logistic.fw(-x)
