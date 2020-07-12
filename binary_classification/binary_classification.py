import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


_min_x = -1
_max_x = 1

_min_y = -1
_max_y = 1


class PointDrawer:

    def __init__(self, pointsA, pointsB, drawing_func):
        self._pos_points = pointsA
        self._beg_points = pointsB
        self._drawing_func = drawing_func

        self._x1_pos = list(self._pos_points.get_xdata())
        self._x2_pos = list(self._pos_points.get_ydata())

        self._x1_neg = list(self._beg_points.get_xdata())
        self._x2_neg = list(self._beg_points.get_ydata())

        self._canvas = self._pos_points.figure.canvas
        self._ax = self._pos_points.axes

        # Assign the callback to some events.
        self._click = self._canvas.mpl_connect('button_press_event', self)
        self._press = self._canvas.mpl_connect('key_release_event', self)

    def _format_data_points(self):
        x1 = np.r_[self._x1_pos, self._x1_neg]
        x2 = np.r_[self._x2_pos, self._x2_neg]
        xs = np.c_[x1, x2]
        ys = np.r_[np.ones((len(self._x1_pos),)),
                   -np.ones((len(self._x1_neg),))]

        return xs, ys

    def __call__(self, event):
        if event.name == 'key_release_event':
            if event.key != 'enter':
                return

            if len(self._x1_pos) < 2:
                # We need at least 2 points.
                return

            xs, ys = self._format_data_points()
            self._drawing_func(self._ax, xs, ys)

        elif event.name == 'button_press_event':
            if event.inaxes != self._pos_points.axes:
                return

            if event.button == 1:
                # Left-click.
                self._x1_pos.append(event.xdata)
                self._x2_pos.append(event.ydata)
                self._pos_points.set_data(self._x1_pos, self._x2_pos)

            elif event.button == 3:
                # Right-click.
                self._x1_neg.append(event.xdata)
                self._x2_neg.append(event.ydata)
                self._beg_points.set_data(self._x1_neg, self._x2_neg)

        self._canvas.draw()


def _linear_kernel(xs):
    return np.array(xs)


def _polynomial_kernel(xs):
    xs = np.array(xs)
    x1 = xs[:, 0]
    x2 = xs[:, 1]

    phis = np.zeros((len(xs), 6))

    phis[:, 0] = np.ones((len(xs),))
    phis[:, 1] = x1
    phis[:, 2] = x2
    phis[:, 3] = x1 * x2
    phis[:, 4] = x1 ** 2
    phis[:, 5] = x2 ** 2

    return phis


def _least_squares(xs, ys):
    lambd = 0

    xs = np.c_[_linear_kernel(xs), np.ones(len(xs))]
    # xs = _polynomial_kernel(xs)
    N, d = xs.shape

    params_w = np.linalg.inv(xs.T @ xs + lambd * np.eye(d)) @ xs.T @ ys
    return params_w


def _perceptron(xs, ys):
    max_iterations = 1000

    xs = np.c_[_linear_kernel(xs), np.ones(len(xs))]
    N, d = xs.shape

    params = np.random.randn(d)
    iter_idx = 0
    while True:
        # Count how many misclassified examples:
        scores = xs @ params
        misclassified = (scores * ys) <= 0
        num_misclassified = np.sum(misclassified)
        print('{} misclassified'.format(num_misclassified))
        if num_misclassified == 0:
            break

        # Pick the next sample.
        idx = iter_idx % N
        if misclassified[idx]:
            # If it was misclassified, use it to update the weights.
            params = params + xs[idx] * ys[idx]

        iter_idx += 1
        if iter_idx >= max_iterations:
            raise ValueError('Impossible solution: sure that points are '
                             'linearly separable?')

    return params


def _draw_prediction(ax, x_coords, y_coords):
    # params_w = _least_squares(x_coords, y_coords)
    params_w = _perceptron(x_coords, y_coords)

    # Create a grid.
    x = np.linspace(_min_x, _max_x, 101)
    x1, x2 = np.meshgrid(x, x)

    # Flatten the coordinates.
    xs = np.c_[np.ravel(x1), np.ravel(x2)]

    # Predict the lables.
    phis = np.c_[_linear_kernel(xs), np.ones(len(xs))]
    # phis = _polynomial_kernel(xs)
    ys = phis @ params_w

    # Reshape back as a grid.
    ys = ys.reshape(x1.shape)

    # Draw the contour regions.
    ax.contourf(x, x, ys,
                # cmap=cm.get_cmap('bwr'),
                levels=(-np.inf, 0, np.inf),
                colors=('b', 'r'),
                alpha=0.2)

    # Draw the boundary.
    ax.contour(x, x, ys,
               levels=(0,),
               colors=('k',),
               linestyles=('solid',))


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Ridge Regression\n'
                 'Left-click for red dots,\n'
                 'Right-click for blue crosses')

    ax.set_xlim([_min_x, _max_x])
    ax.set_ylim([_min_y, _max_y])
    ax.set_aspect('equal')

    red_dots, = ax.plot([], [], linestyle='none', marker='o', color='r')
    blue_crosses, = ax.plot([], [], linestyle='none', marker='x', color='b')

    point_drawer = PointDrawer(red_dots, blue_crosses, _draw_prediction)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
