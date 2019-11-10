import numpy as np


def evaluate_accuracy(predictions, labels):
    accuracy = np.mean(predictions == labels)
    return accuracy


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
    classes = np.unique(labels)

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


def compute_priors(labels, weights=None):
    """Computes the priors on the labels.

    In this simple implementation the prior is computed as the frequency
    each label appears.

    Args:
        labels: Shape=(N,); label for each sample.
        weights: TODO

    Returns:
        A vector of probabilities with shape=(C,) with C number of classes.
    """
    N = len(labels)
    if weights is None:
        weights = np.ones(N) / N
    else:
        assert (len(weights) == N)

    classes = np.unique(labels)
    priors = np.array([len(np.where(labels == class_id)[0])
                       for class_id in classes]) / N
    return priors


def classify_bayes(samples, priors, mu, sigma):
    """Uses the Bayes Theorem to classify the provided samples.

    Args:
        samples: Shape=(N, D); samples to classify, where N is the number of
            samples and D their dimensionality.
        priors: Shape=(C,); priors on the labels, with C number of classes.
        mu: Shape=(C, D); estimated means, for each class.
        sigma: Shape=(C, D, D); estimated covariance matrices, for each class.

    Returns:
        A vector with shape=(N,) with the class prediction for each sample.
    """
    N = samples.shape[0]
    C, D = mu.shape

    def _log_prob(samples_, mu_, sigma_, prior_):
        """Computes the logarithm of the Gaussian probability of
        the provided samples.

        Args:
            samples_: Shape=(N, D).
            mu_: Shape=(D,).
            sigma_: Shape=(D, D).
            prior_: Scalar.

        Returns:
            A vector with shape=(N,) with the log-probabilities of the N
            input samples..
        """
        aux1 = -0.5 * np.log(np.linalg.det(sigma_))      # scalar
        aux2 = samples_ - mu_[np.newaxis, :]            # (N, D)
        sigma_inv = np.linalg.inv(sigma_)               # (D, D)
        aux3 = aux2 @ sigma_inv                         # (N, D)
        aux4 = aux3 * aux2                              # (N, D)
        aux5 = -0.5 * np.sum(aux4, axis=1)              # (N,)
        log_prob_ = aux1 + aux5 + np.log(prior_)
        return log_prob_

    # Generate a (N, C) matrix.
    log_probs = np.array([
        _log_prob(samples_=samples,
                  mu_=mu[class_idx],
                  sigma_=sigma[class_idx],
                  prior_=priors[class_idx])
        for class_idx in range(C)
    ]).T

    assert log_probs.shape == (N, C)

    predictions = np.argmax(log_probs, axis=1)
    return predictions


class BayesClassifier:

    def __init__(self, priors, mu, sigma):
        self._priors = priors
        self._mu = mu
        self._sigma = sigma

        self.labels = []

    def classify(self, samples):
        predictions = classify_bayes(samples=samples,
                                     priors=self._priors,
                                     mu=self._mu,
                                     sigma=self._sigma)
        return predictions

    @staticmethod
    def train(samples, labels, naive):
        priors = compute_priors(labels=labels)
        mu, sigma = maximum_likelihood_estimator(samples=samples,
                                                 labels=labels,
                                                 naive=naive)
        bayes_classifier = BayesClassifier(priors=priors,
                                           mu=mu,
                                           sigma=sigma)
        bayes_classifier.labels = np.unique(labels)
        return bayes_classifier

    @property
    def priors(self):
        return self._priors

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma
