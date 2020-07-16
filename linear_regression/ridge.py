import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import linear_regression.point_drawer as drawer

sns.set()


_min_x = -5
_max_x = 5

_min_y = - 2
_max_y = 2

# Regularization parameters.
_lambdas = [0.0, 0.1, 1.0, 10.0]


def _ridge_regression(xs, ys, lambd):
    # Append a 1 to each data point to learn the bias.
    xs = np.c_[xs, np.ones((len(xs),))]

    dims = xs.shape[1]
    W = np.linalg.inv(xs.T @ xs + lambd * np.eye(dims)) @ xs.T @ ys

    # This std deviation assumes that the underlying model is correct and the
    # noise on the labels is Gaussian.
    sigma = np.sqrt(np.sum((xs @ W - ys) ** 2))

    return W, sigma


def _draw_prediction(ax, xs, ys):
    ys = np.atleast_2d(ys)
    zs = np.linspace(_min_x, _max_x, 1001)

    # Append a 1 to each data point to account for the bias.
    zzs = np.expand_dims(np.linspace(_min_x, _max_x, 1001), axis=1)
    zzs = np.c_[zzs, np.ones(len(zzs), )]

    for lambda_reg in _lambdas:
        W, sigma = _ridge_regression(xs, ys, lambda_reg)
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
