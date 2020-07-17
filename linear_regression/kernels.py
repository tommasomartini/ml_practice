import numpy as np


def _kernel(func):
    def _wrapper(xs, zs=None, **kwargs):
        xs = np.atleast_2d(xs)
        if zs is None:
            zs = xs
        else:
            zs = np.atleast_2d(zs)
        return func(xs, zs, **kwargs)
    return _wrapper


@_kernel
def linear(xs, zs=None):
    return xs @ zs.T


@_kernel
def polynomial(xs, zs=None, degree=2, offset=1):
    N = xs.shape[0]
    M = zs.shape[0]
    K = offset * np.ones((N, M))
    K = (K + xs @ zs.T) ** degree
    return K


@_kernel
def rbf(xs, zs=None, sigma=1.0):
    N = xs.shape[0]
    M = zs.shape[0]

    xxs = np.repeat(np.expand_dims(xs, axis=2), M, axis=2)
    zzs = np.repeat(np.expand_dims(zs.T, axis=0), N, axis=0)

    exp_num = np.linalg.norm(xxs - zzs, axis=1)
    exp_den = sigma ** 2

    K = np.exp(- exp_num / exp_den)

    return K
