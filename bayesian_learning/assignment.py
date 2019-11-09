import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs

import bayesian_learning.bayesian_learning_utils as bl
import bayesian_learning.plotting as plotting

sns.set()


def _separator():
    sep_line = '#' * 80
    print()
    print(sep_line)
    print()


def assignment1():
    print('Assignment 1')
    print('Compute the Maximum Likelihood estimation '
          'of a synthetic Gaussian dataset.')

    # Generate a dataset.
    samples, labels = make_blobs(n_samples=200,
                                 centers=5,
                                 n_features=2,
                                 random_state=0)
    mu, sigma = bl.maximum_likelihood_estimator(samples, labels, naive=False)

    plt.figure()
    ax = plt.gca()
    plotting.plot_gaussian(ax=ax,
                           samples=samples,
                           labels=labels,
                           mu=mu,
                           sigma=sigma)
    plt.title('Assignment 1')
    ax.legend()
    plt.show()
    plt.close()


def assignment2():
    print('Assignment 2')
    print('Compute the prior and classify using the Bayesian rule.')

    # Generate a dataset.
    samples, labels = make_blobs(n_samples=200,
                                 centers=5,
                                 n_features=2,
                                 random_state=0)
    mu, sigma = bl.maximum_likelihood_estimator(samples, labels, naive=False)
    priors = bl.compute_priors(labels)
    predictions = bl.classify_bayes(samples=samples,
                                    priors=priors,
                                    mu=mu,
                                    sigma=sigma)

    accuracy = bl.evaluate_accuracy(predictions, labels)
    print('Accuracy: {:.3f}'.format(accuracy))


def main():
    # assignment1()
    # _separator()
    assignment2()


if __name__ == '__main__':
    main()
