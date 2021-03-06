import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import common.point_drawer as drawer

sns.set()

_EPS = 1e-8


_min_x = - 5
_max_x = 5

_min_y = - 2
_max_y = 2

# Regularization parameters.
_lambdas = [0.0, 0.1, 1.0, 10.0, 100.0]

_lr = 0.01
_n_iters = 1000


def _lasso(xs, ys, lambd):
    # Append a 1 to each data point to learn the bias.
    xs = np.c_[xs, np.ones((len(xs),))]

    dims = xs.shape[1]

    W = np.random.randn(dims, 1)
    for iter_idx in range(_n_iters):
        loss = np.mean((ys - xs @ W) ** 2)
        # print('Iter {}, loss={:6.3f}'.format(iter_idx, loss))

        dL_dW = - xs.T @ ys + xs.T @ xs @ W + lambd * W / (np.abs(W) + _EPS)
        W = W - _lr * dL_dW

    sigma = np.sqrt(np.sum((xs @ W - ys) ** 2))

    return W, sigma


def _draw_prediction(ax, xs, ys):
    ys = np.atleast_2d(ys)
    zs = np.linspace(_min_x, _max_x, 1001)

    # Append a 1 to each data point to account for the bias.
    zzs = np.expand_dims(zs, axis=1)
    zzs = np.c_[zzs, np.ones(len(zzs), )]

    for lambda_reg in _lambdas:
        W, sigma = _lasso(xs, ys, lambda_reg)

        print('Lambda={:.2f} -> W=[{:.2f}, {:.2f}]'.format(lambda_reg,
                                                           W[0, 0],
                                                           W[1, 0]))

        ys_hat = zzs @ W

        # Plot the mean.
        ax.plot(zs, ys_hat, label='lambda={:.1f}'.format(lambda_reg))

        # # Plot the sigma boundaries.
        # ax.fill_between(np.squeeze(zs), np.squeeze(ys_hat - sigma), np.squeeze(ys_hat + sigma),
        #                 alpha=0.5)

    plt.legend()


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Left-click for red dots,\n'
                 'Enter to predict')

    ax.set_xlim([_min_x, _max_x])
    ax.set_ylim([_min_y, _max_y])

    red_dots, = ax.plot([], [], linestyle='none', marker='o', color='r')

    drawer.PointDrawer1D(red_dots, _draw_prediction)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
