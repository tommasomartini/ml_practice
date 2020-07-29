import numpy as np


class SGD:

    def update(self, parameters, gradients, learning_rate):
        new_parameters = parameters - learning_rate * gradients
        return new_parameters


class Adam:

    _EPS = 1e-8

    def __init__(self, beta1=0.9, beta2=0.999):
        self._beta1 = beta1
        self._beta2 = beta2

        self._iter = 0
        self._prev_v = 0
        self._prev_s = 0

    def update(self, parameters, gradients, learning_rate):
        self._iter += 1

        v = self._beta1 * self._prev_v + (1 - self._beta1) * gradients
        s = self._beta2 * self._prev_s + (1 - self._beta2) * (gradients ** 2)

        v_star = v / (1 - self._beta1 ** self._iter)
        s_star = s / (1 - self._beta2 ** self._iter)

        update_factor = v_star / (np.sqrt(s_star) + self._EPS)
        new_parameters = parameters - learning_rate * update_factor

        self._prev_v = v
        self._prev_s = s

        return new_parameters
