import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import bayesian_learning.plotting as plotting


def _covariance(samples):
    """Computes the covariance matrix of the input samples.

    This implementation consider the `samples` matrix as a N realizations of
    D variables. Hence the covariance matrix should have shape (D, D).

    This implementation uses the unbiased version of covariance computation,
    hence dividing by (N - 1) instead of N.

    Args:
        samples: Shape=(N, D) with N number of samples and D dimensionality.

    Returns:
        The covariance matrix, shape=(D, D).
    """
    N, D = samples.shape
    mu = np.mean(samples, axis=0)
    cov = (samples - mu).T @ (samples - mu) / (N - 1)

    assert cov.shape == (D, D)
    np.testing.assert_array_almost_equal(cov, np.cov(samples, rowvar=False))

    return cov


def maximum_likelihood_estimator(samples, labels, weights=None, naive=True):
    """Estimates the mean vector and the covariance matrix of a dataset assuming
    an underlying Gaussian distribution.

    Args:
        samples: Shape=(N, D) with N number of samples and D dimensionality.
        labels: Shape=(N,) with N number of samples.
        weights: TODO
        naive (bool, optional): If True, the Naive Bayes assumption is applied
            and the covariance matrix is created diagonal.

    Returns:
        A tuple (mu, sigma) with:
        mu: shape=(C, D) vector of mean values for each class.
        sigma: shape=(C, D, D) vector of covariance matrices for each class.
        C is the number of classes and D sample dimensionality.
    """
    assert samples.shape[0] == labels.shape[0]

    N, D = samples.shape
    classes = sorted(list(set(labels)))

    if weights is None:
        weights = np.ones(N) / N

    def _compute_mu_and_sigma_for_class(class_id):
        class_indices = np.where(labels == class_id)[0]
        class_samples = samples[class_indices, :]

        class_mean = np.mean(class_samples, axis=0)

        class_covariance = _covariance(class_samples)
        if naive:
            # Apply the Naive Bayes assumption.
            class_covariance = np.diag(np.diag(class_covariance))

        return class_mean, class_covariance

    mu_sigma = [
        _compute_mu_and_sigma_for_class(class_id)
        for class_id in classes
    ]
    mu, sigma = map(np.array, zip(*mu_sigma))

    return mu, sigma


def assignment1():
    print('Assignment 1')
    print('Compute the Maximum Likelihood estimation '
          'of a synthetic Gaussian dataset.')

    # Generate a dataset.
    samples, labels = make_blobs(n_samples=200,
                                 centers=5,
                                 n_features=2,
                                 random_state=0)
    mu, sigma = maximum_likelihood_estimator(samples, labels, naive=False)
    plotting.plotGaussian(samples, labels, mu, sigma)


def main():
    assignment1()


if __name__ == '__main__':
    main()
