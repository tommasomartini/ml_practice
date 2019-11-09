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
    ax.add_artist(ellipse)


def plot_gaussian(ax, samples, labels, mu, sigma):
    classes = sorted(list(set(labels)))
    cmap = cm.get_cmap('tab10')
    for idx, class_id in enumerate(classes):
        # class_color = cmap(idx / (len(classes) - 1))
        class_color = cmap((idx % 10) / 10)

        # Draw the samples.
        class_indices = np.where(labels == class_id)[0]
        class_samples = samples[class_indices]
        ax.scatter(class_samples[:, 0],
                   class_samples[:, 1],
                   color=class_color,
                   marker='o',
                   label='Class {}'.format(class_id))

        # Draw the estimated Gaussian contour.
        class_mu = mu[idx]
        class_sigma = sigma[idx]
        plot_gaussian_distribution_ellipse(ax=ax,
                                           mu=class_mu,
                                           sigma=class_sigma,
                                           color=class_color,
                                           alpha=0.3)
