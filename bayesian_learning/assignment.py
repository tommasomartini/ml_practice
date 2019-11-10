import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs

import bayesian_learning.bayesian_classifier as bc
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
    mu, sigma = bc.maximum_likelihood_estimator(samples, labels, naive=False)

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

    dataset_size = 200

    # Generate a dataset.
    samples, labels = make_blobs(n_samples=2 * dataset_size,
                                 centers=5,
                                 n_features=2,
                                 random_state=0)

    training_samples = samples[:dataset_size]
    test_samples = samples[dataset_size:]

    training_labels = labels[:dataset_size]
    test_labels = labels[dataset_size:]

    bayes_classifier = bc.BayesClassifier.train(samples=training_samples,
                                                labels=training_labels,
                                                naive=False)

    # Evaluate on the training data.
    training_predictions = bayes_classifier.classify(training_samples)
    training_accuracy = bc.evaluate_accuracy(predictions=training_predictions,
                                             labels=training_labels)
    print('Training accuracy: {:.3f}'.format(training_accuracy))

    # Evaluate on the test data.
    test_predictions = bayes_classifier.classify(test_samples)
    test_accuracy = bc.evaluate_accuracy(predictions=test_predictions,
                                         labels=test_labels)
    print('Test accuracy: {:.3f}'.format(test_accuracy))

    # Plot the classification of the test samples.
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plotting.plot_gaussian(ax=ax1,
                           samples=test_samples,
                           labels=test_predictions,
                           mu=bayes_classifier.mu,
                           sigma=bayes_classifier.sigma)
    plotting.plot_boudaries(ax=ax1,
                            classifier=bayes_classifier,
                            grid_size=1000)
    ax1.legend()
    ax1.set_title('Prediction')

    plotting.plot_gaussian(ax=ax2,
                           samples=test_samples,
                           labels=test_labels,
                           mu=bayes_classifier.mu,
                           sigma=bayes_classifier.sigma)
    ax2.legend()
    ax2.set_title('Ground truth')

    fig.suptitle('Assignment 2 - Test samples\n'
                 'Accuracy: {:.3f}'.format(test_accuracy))
    plt.show()
    plt.close()


def assignment3():
    pass


def main():
    # assignment1()
    # _separator()
    assignment2()
    # _separator()
    # assignment3()


if __name__ == '__main__':
    main()
