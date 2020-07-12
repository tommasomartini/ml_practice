import numpy as np
from matplotlib import pyplot as plt


_min_x = 0
_max_x = 5

_min_y = - 2
_max_y = 2


class PointDrawer:

    def __init__(self, pointsA, pointsB, drawing_func):
        self._pointsA = pointsA
        self._pointsB = pointsB
        self._drawing_func = drawing_func

        self._xA = list(self._pointsA.get_xdata())
        self._yA = list(self._pointsA.get_ydata())

        self._xB = list(self._pointsB.get_xdata())
        self._yB = list(self._pointsB.get_ydata())

        self._canvas = self._pointsA.figure.canvas
        self._ax = self._pointsA.axes

        # Assign the callback to some events.
        self._click = self._canvas.mpl_connect('button_press_event', self)
        self._press = self._canvas.mpl_connect('key_release_event', self)

    def __call__(self, event):
        if event.name == 'key_release_event':
            if event.key != 'enter':
                return

            if len(self._xA) < 2:
                # We need at least 2 points.
                return

            self._drawing_func(self._ax, self._xA, self._yA)

        elif event.name == 'button_press_event':
            if event.inaxes != self._pointsA.axes:
                return

            if event.button == 1:
                # Left-click.
                self._xA.append(event.xdata)
                self._yA.append(event.ydata)
                self._pointsA.set_data(self._xA, self._yA)

            elif event.button == 3:
                # Right-click.
                self._xB.append(event.xdata)
                self._yB.append(event.ydata)
                self._pointsB.set_data(self._xB, self._yB)

        self._canvas.draw()


def _polynomial_kernel(xs, degree=2):
    xs = np.array(xs)
    phis = np.zeros((len(xs), degree + 1))
    for idx in range(degree + 1):
        phis[:, idx] = xs ** idx
    return phis


def _rbf_kernel(xs, mu=0.0, sigma=1.0):
    return


def _closed_form_linear_regression(x_coords, y_coords):
    phis = _polynomial_kernel(x_coords)
    Y = np.expand_dims(np.array(y_coords), axis=1)

    dims = phis.shape[1]

    lambd = 1.0
    params_w = np.linalg.inv(phis.T @ phis + lambd * np.eye(dims)) @ phis.T @ Y

    # This std deviation assumes that the underlying model is correct and the
    # noise on the labels is Gaussian.
    sigma = np.sqrt(np.sum((phis @ params_w - Y) ** 2))

    return params_w, sigma


def _iterative_least_squares(x_coords, y_coords):
    lr = 0.0001
    n_iters = 100
    lambd = 1.0

    phis = _polynomial_kernel(x_coords)
    Y = np.expand_dims(np.array(y_coords), axis=1)

    dims = phis.shape[1]
    params_w = np.zeros((dims,1))

    for iter_idx in range(n_iters):
        dL_dW = - phis.T @ Y + (phis.T @ phis + lambd * np.eye(dims)) @ params_w
        params_w = params_w - lr * dL_dW

    sigma = np.sqrt(np.sum((phis @ params_w - Y) ** 2))

    return params_w, sigma


def _ridge_regression(x_coords, y_coords, closed_form=True):
    if closed_form:
        return _closed_form_linear_regression(x_coords, y_coords)

    else:
        return _iterative_least_squares(x_coords, y_coords)


def _lasso(x_coords, y_coords):
    lr = 0.01
    n_iters = 100
    lambd = 0.0
    eps = 1e-8

    phis = _polynomial_kernel(x_coords)
    Y = np.expand_dims(np.array(y_coords), axis=1)

    dims = phis.shape[1]
    params_w = np.zeros((dims,1))

    for iter_idx in range(n_iters):
        dL_dW = - phis.T @ Y + phis.T @ phis @ params_w + lambd * params_w / (params_w + eps)
        params_w = params_w - lr * dL_dW

    sigma = np.sqrt(np.sum((phis @ params_w - Y) ** 2))

    return params_w, sigma


def _draw_prediction(ax, x_coords, y_coords):
    # params_w, sigma = _ridge_regression(x_coords, y_coords)
    params_w, sigma = _lasso(x_coords, y_coords)

    # Append a 1 to each data point to account for the bias.
    xs = np.linspace(_min_x, _max_x, 1001)
    phis = _polynomial_kernel(xs)
    ys = phis @ params_w

    # Plot the mean.
    ax.plot(xs, ys)

    # Plot the sigma boundaries.
    ax.fill_between(xs, np.squeeze(ys - sigma), np.squeeze(ys + sigma), alpha=0.5)


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Ridge Regression\n'
                 'Left-click for red dots,\n'
                 'Right-click for blue crosses')

    ax.set_xlim([_min_x, _max_x])
    ax.set_ylim([_min_y, _max_y])

    red_dots, = ax.plot([], [], linestyle='none', marker='o', color='r')
    blue_crosses, = ax.plot([], [], linestyle='none', marker='x', color='b')

    point_drawer = PointDrawer(red_dots, blue_crosses, _draw_prediction)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
