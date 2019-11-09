import numpy as np

_RANGE_SCALE = 1.2
_DEFAULT_GRID_SIZE = (100, 100)


def plot_data_points(ax, dataset):
    pos_samples = dataset[np.where(dataset[:, 2] == 1)]
    neg_samples = dataset[np.where(dataset[:, 2] == -1)]

    assert len(pos_samples) + len(neg_samples) == len(dataset)

    ax.scatter(pos_samples[:, 0],
               pos_samples[:, 1],
               color='b',
               marker='o',
               label='Positive')
    ax.scatter(neg_samples[:, 0],
               neg_samples[:, 1],
               color='g',
               marker='x',
               label='Negative')


def plot_margins(ax, indicator_function, xlims, ylims, grid_size=None):
    try:
        min_x, max_x = xlims
    except:
        min_x, max_x = -xlims, xlims

    try:
        min_y, max_y = ylims
    except:
        min_y, max_y = -ylims, ylims

    grid_size = grid_size or _DEFAULT_GRID_SIZE
    try:
        grid_size_x, grid_size_y = grid_size
    except:
        grid_size_x, grid_size_y = grid_size, grid_size

    xrange = np.linspace(min_x, max_x, grid_size_x)
    yrange = np.linspace(min_y, max_y, grid_size_y)
    xx, yy = np.meshgrid(xrange, yrange, indexing='xy')
    zz = indicator_function(np.c_[xx.ravel(), yy.ravel()])
    grid = zz.reshape(xx.shape)

    contours = ax.contour(xrange,
                          yrange,
                          grid,
                          levels=(-1, 0, 1),
                          colors=('g', 'k', 'b'),
                          linestyles=('dashed', 'solid', 'dashed'))
    ax.clabel(contours, fontsize=10)


def plot_data_points_and_margin(ax,
                                dataset,
                                indicator_function,
                                grid_size=None):
    x_coords = dataset[:, 0]
    y_coords = dataset[:, 1]

    def _get_axis_lims(coords):
        min_coord = np.min(coords)
        max_coord = np.max(coords)
        mid_point = 0.5 * (min_coord + max_coord)
        lims = [
            mid_point - _RANGE_SCALE * (max_coord - min_coord) / 2,
            mid_point + _RANGE_SCALE * (max_coord - min_coord) / 2,
            ]
        return lims

    xlims = _get_axis_lims(x_coords)
    ylims = _get_axis_lims(y_coords)

    plot_data_points(ax, dataset)
    plot_margins(ax, indicator_function, xlims, ylims, grid_size)


def _tryout():
    import matplotlib.pyplot as plt

    plt.figure()
    ax = plt.gca()

    def _indic_func(*coords):
        xx, yy = coords
        zz1 = np.exp(-xx ** 2 - yy ** 2)
        zz2 = np.exp(-(xx - 1) ** 2 - (yy - 1) ** 2)
        zz = (zz1 - zz2) * 2
        return zz

    plot_margins(ax,
                 indicator_function=_indic_func,
                 xlims=[-3, 3],
                 ylims=[-2, 2])

    plt.show()
    plt.close()


if __name__ == '__main__':
    _tryout()
