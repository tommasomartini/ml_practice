import numpy as np


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

            # These Axes contains a number of lines: remove them all, but the
            # first one, which is the input points.
            self._ax.lines = self._ax.lines[:1]

            xs = np.array([self._xA]).T
            ys = np.array([self._yA]).T
            self._drawing_func(self._ax, xs, ys)

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