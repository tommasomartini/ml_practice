import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Ellipse


_NUM_STD_FOR_GAUSSIAN_PLOT = 2


def plot_gaussian_distribution_ellipse(ax, mu, sigma, **kwargs):
    """In this implementation we assume samples are 2D.
    """

    def _sorted_eigenvalues(matrix_):
        """Returns the eigenvalues and the eigenvectors of the input matrix
        in descending order.

        Returns a pair (sorted_eigenvals, sorted_eigenvecs) where:
            sorted_eigenvals: shape=(D,)
            sorted_eigenvecs: shape=(D, D)
        given that `matrix_` has shape=(D, D).
        """
        # Eigenvectors are in columns.
        eigenvals_, eigenvecs_ = np.linalg.eigh(matrix_)
        descending_indices = np.argsort(eigenvals_)[::-1]
        sorted_eigenvals = eigenvals_[descending_indices]
        sorted_eigenvecs = eigenvecs_.T[descending_indices, :]
        return sorted_eigenvals, sorted_eigenvecs

    # Infer the orientation of the Gaussian bell.
    # Intuitively speaking, the eigenvectors of a covariance matrix are
    # the directions along which the data vary the most.
    # We want to rotate the ellipse along the direction of largest variation.
    eigenvals, eigenvecs = _sorted_eigenvalues(sigma)   # descending order

    first_eigenvector = eigenvecs[0]
    eigenvec_x = first_eigenvector[0]
    eigenvec_y = first_eigenvector[1]
    theta_rad = np.arctan2(eigenvec_y, eigenvec_x)
    theta_deg = np.rad2deg(theta_rad)

    # Width and height are "full" widths, not radii.
    width, height = 2 * _NUM_STD_FOR_GAUSSIAN_PLOT * np.sqrt(eigenvals)
    ellipse = Ellipse(xy=mu,
                      width=width,
                      height=height,
                      angle=theta_deg,
                      **kwargs)
    ellipse.set_alpha(0.25)
    ax.add_artist(ellipse)


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ellip.set_alpha(0.25)

    ax.add_artist(ellip)
    return ellip


def plotGaussian(X,y,mu,sigma):
    labels = np.unique(y)
    Ncolors = len(labels)
    xx = np.arange(Ncolors)
    ys = [i+xx+(i*xx)**2 for i in range(Ncolors)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = 1.0
    for label in labels:
        classIdx = y==label
        Xclass = X[classIdx,:]
        # plot_gaussian_distribution_ellipse(ax=plt.gca(),
        #                                    mu=mu[label],
        #                                    sigma=sigma[label])
        plot_cov_ellipse(sigma[label], mu[label])
        plt.scatter(Xclass[:,0],Xclass[:,1],linewidths=1,s=25,color=colors[label],marker='o',alpha=0.75)
        c += 1.

    plt.show()


def _tryout():
    plt.figure()

    plt.show()
    plt.close()


if __name__ == '__main__':
    _tryout()
