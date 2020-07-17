import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import linear_regression.kernels as kernels
import linear_regression.point_drawer as drawer

sns.set()


_min_x = -5
_max_x = 5

_min_y = - 2
_max_y = 2

# Regularization parameters.
_lambdas = [0.1, 1.0, 10.0, 100.0]

_kernel = kernels.rbf
_kernel_kwargs = {'sigma': 1.}


def _ridge_regression(xs, ys, lambd):
    K = _kernel(xs=xs, **_kernel_kwargs)
    alphas = np.linalg.inv(K + lambd * np.eye(len(K))) @ ys
    return alphas


def _draw_prediction(ax, xs, ys):
    ys = np.atleast_2d(ys)
    zs = np.linspace(_min_x, _max_x, 1001)

    zs = np.expand_dims(zs, axis=1)
    for lambda_reg in _lambdas:
        alphas = _ridge_regression(xs, ys, lambda_reg)
        K = _kernel(xs=xs, zs=zs, **_kernel_kwargs)
        ys_hat = K.T @ alphas

        # Plot the mean.
        ax.plot(zs, ys_hat, label='lambda={:.1f}'.format(lambda_reg))

    plt.legend()


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Left-click for red dots,\n'
                 'Right-click for blue crosses\n'
                 'Enter to predict')

    ax.set_xlim([_min_x, _max_x])
    ax.set_ylim([_min_y, _max_y])

    red_dots, = ax.plot([], [], linestyle='none', marker='o', color='r')
    blue_crosses, = ax.plot([], [], linestyle='none', marker='x', color='b')

    drawer.PointDrawer(red_dots, blue_crosses, _draw_prediction)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
