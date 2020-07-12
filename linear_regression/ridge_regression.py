import numpy as np
from matplotlib import pyplot as plt


_min_x = 0
_max_x = 5


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


def _polynomial_kernel(xs, degree=5):
    xs = np.array(xs)
    phis = np.zeros((len(xs), degree + 1))
    for idx in range(degree + 1):
        phis[:, idx] = xs ** idx
    return phis


def _ridge_regression(x_coords, y_coords):
    # Add a "1" component to each data point, effectively expanding by one unit
    # their dimensions, so that we can account for a bias.
    X = _polynomial_kernel(x_coords)
    Y = np.expand_dims(np.array(y_coords), axis=1)

    dims = X.shape[1]

    lambd = 1.0
    params_w = np.linalg.inv(X.T @ X + lambd * np.eye(dims)) @ X.T @ Y

    return params_w


def _draw_prediction(ax, x_coords, y_coords):
    params_w = _ridge_regression(x_coords, y_coords)

    # Append a 1 to each data point to account for the bias.
    xs = np.linspace(_min_x, _max_x, 21)
    phis = _polynomial_kernel(xs)
    ys = phis @ params_w
    ax.plot(xs, ys)


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Ridge Regression\n'
                 'Left-click for red dots,\n'
                 'Right-click for blue crosses')

    ax.set_xlim([_min_x, _max_x])
    ax.set_ylim([0, 1])

    red_dots, = ax.plot([], [], linestyle='none', marker='o', color='r')
    blue_crosses, = ax.plot([], [], linestyle='none', marker='x', color='b')

    point_drawer = PointDrawer(red_dots, blue_crosses, _draw_prediction)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
