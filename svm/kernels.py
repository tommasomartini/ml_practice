import numpy as np


def get_linear_kernel():
    def _ker(v1, v2):
        v1, v2 = np.atleast_2d(v1, v2)
        res = np.squeeze(v1 @ v2.T + 1)
        return res

    return _ker


def get_polynomial_kernel(p):
    def _ker(v1, v2):
        v1, v2 = np.atleast_2d(v1, v2)
        res = np.squeeze(np.power((v1 @ v2.T + 1), p))
        return res

    return _ker
