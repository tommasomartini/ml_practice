"""This module contains a collection of kernel function generators.
Each function returns a kernel function, which has the following
signature:
    res = _ker(v1, v2)
with:
    v1, v2: Array-like with shape=(N, D) or (D,) where N is the number of points
        and D is the dimensionality of each point.

    res: Scalar or array-like with shape=(N,).
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
        aux1 = np.linalg.norm(v1 - v2, axis=1) ** 2
        aux2 = - aux1 / sigma ** 2
        aux3 = np.exp(aux2)
        res = np.squeeze(aux3)
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
