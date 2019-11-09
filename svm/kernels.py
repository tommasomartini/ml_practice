"""This module contains a collection of kernel function generators.
Each function returns a kernel function, which has the following
signature:
    res = _ker(v1, v2)
with:
    v1: Array-like with shape=(N1, D) or (D,) where N1 is the number of points
        and D is the dimensionality of each point.
    v2: Array-like with shape=(N2, D) or (D,) where N2 is the number of points
        and D is the dimensionality of each point.

    res: Scalar or array-like with shape=(N1, N2).
"""

import numpy as np


def get_linear_kernel():
    """Implements:
        <x.T, y> + 1
    """
    def _ker(v1, v2):
        v1, v2 = np.atleast_2d(v1, v2)
        res = np.squeeze(v1 @ v2.T + 1)
        return res

    return _ker


def get_polynomial_kernel(p):
    """Implements:
        (<x.T, y> + 1)^p
    """
    def _ker(v1, v2):
        v1, v2 = np.atleast_2d(v1, v2)
        res = np.squeeze(np.power((v1 @ v2.T + 1), p))
        return res

    return _ker


def get_radial_basis_function_kernel(sigma):
    """Implements:
        exp(- ||x - y||^2 / sigma^2)
    """
    def _ker(v1, v2):
        v1, v2 = np.atleast_2d(v1, v2)

        N1, D1 = v1.shape
        N2, D2 = v2.shape
        assert D1 == D2
        D = D1

        v1_r = v1.reshape((N1, 1, D))
        v2_r = v2.reshape((1, N2, D))

        aux1 = v1_r - v2_r                          # (N1, N2, D)
        aux2 = np.linalg.norm(aux1, axis=-1) ** 2   # (N1, N2)
        aux3 = - aux2 / sigma ** 2                  # (N1, N2)
        res = np.squeeze(np.exp(aux3))              # (N1, N2)
        return res

    return _ker


def get_sigmoid_kernel(k, delta):
    """Implements:
        tanh(<k * x.T, y> - delta)
    """
    def _ker(v1, v2):
        v1, v2 = np.atleast_2d(v1, v2)
        res = np.squeeze(np.tanh(k * v1 @ v2.T - delta))
        return res

    return _ker
