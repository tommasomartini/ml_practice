import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prettytable import PrettyTable
from sklearn import decomposition
from sklearn.datasets.samples_generator import make_blobs

import bayesian_learning.bayesian_classifier as bc
import bayesian_learning.dataset as dataset
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
                           labels=test_labels,
                           mu=bayes_classifier.mu,
                           sigma=bayes_classifier.sigma)
    ax1.legend()
    ax1.set_title('Ground truth')

    plotting.plot_gaussian(ax=ax2,
                           samples=test_samples,
                           labels=test_predictions,
                           mu=bayes_classifier.mu,
                           sigma=bayes_classifier.sigma)
    plotting.plot_boudaries(ax=ax2,
                            classifier=bayes_classifier,
                            grid_size=1000)
    ax2.legend()
    ax2.set_title('Prediction')

    fig.suptitle('Assignment 2 - Test samples\n'
                 'Accuracy: {:.3f}'.format(test_accuracy))
    plt.show()
    plt.close()


def assignment3():
    print('Assignment 3')
    print('Test your Bayesian Classifier on real datasets.')

    num_trials = 100

    pretty_table = PrettyTable()
    pretty_table.field_names = ['Dataset', 'Non naive', 'Naive']

    datasets = [
        dataset.DatasetNames.IRIS,
        dataset.DatasetNames.WINE,
        dataset.DatasetNames.VOWEL,
    ]
    for dataset_name in datasets:
        samples, labels = dataset.load_dataset(dataset_name)

        accuracies = []
        naive_accuracies = []
        for trial_idx in range(num_trials):
            (training_samples,
             training_labels,
             test_samples,
             test_labels) = dataset.split_dataset(samples=samples,
                                                  labels=labels,
                                                  train_fraction=0.5,
                                                  balance_classes=True,
                                                  seed=trial_idx)

            # Non-naive classifier.
            classifier = bc.BayesClassifier.train(samples=training_samples,
                                                  labels=training_labels,
                                                  naive=False)
            test_predictions = classifier.classify(samples=test_samples)
            test_accuracy = bc.evaluate_accuracy(predictions=test_predictions,
                                                 labels=test_labels)
            accuracies.append(test_accuracy)

            # Naive classifier.
            naive_classifier = \
                bc.BayesClassifier.train(samples=training_samples,
                                         labels=training_labels,
                                         naive=True)
            naive_test_predictions = \
                naive_classifier.classify(samples=test_samples)
            naive_test_accuracy = bc.evaluate_accuracy(
                predictions=naive_test_predictions, labels=test_labels)
            naive_accuracies.append(naive_test_accuracy)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        naive_mean_accuracy = np.mean(naive_accuracies)
        naive_std_accuracy = np.std(naive_accuracies)

        def _format_acc(mean_, std_):
            acc_str = '{:.3f} +/- {:.3f}'.format(mean_, std_)
            return acc_str

        pretty_table.add_row([dataset_name.value,
                              _format_acc(mean_accuracy, std_accuracy),
                              _format_acc(naive_mean_accuracy,
                                          naive_std_accuracy)])

    print()
    print('Mean accuracy on test set ({} trials)'.format(num_trials))
    print(pretty_table)


def assignment3p1():
    print('Assignment 3.1')
    print('Use PCA to reduce the datasets to 2D and plot the boundaries.')

    for dataset_name in dataset.DatasetNames:
        samples, labels = dataset.load_dataset(dataset_name)
        training_samples, training_labels, test_samples, test_labels = \
            dataset.split_dataset(samples=samples,
                                  labels=labels,
                                  train_fraction=0.5,
                                  balance_classes=True,
                                  seed=0)

        pca = decomposition.PCA(n_components=2)
        pca.fit(training_samples)
        training_samples = pca.transform(training_samples)
        test_samples = pca.transform(test_samples)

        classifier = bc.BayesClassifier.train(samples=training_samples,
                                              labels=training_labels,
                                              naive=False)
        test_predictions = classifier.classify(samples=test_samples)
        test_accuracy = bc.evaluate_accuracy(predictions=test_predictions,
                                             labels=test_labels)

        # Plot the classification of the test samples.
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plotting.plot_gaussian(ax=ax1,
                               samples=test_samples,
                               labels=test_labels,
                               mu=classifier.mu,
                               sigma=classifier.sigma)
        ax1.legend()
        ax1.set_title('Ground truth')

        plotting.plot_gaussian(ax=ax2,
                               samples=test_samples,
                               labels=test_predictions,
                               mu=classifier.mu,
                               sigma=classifier.sigma)
        plotting.plot_boudaries(ax=ax2,
                                classifier=classifier,
                                grid_size=1000)
        ax2.legend()
        ax2.set_title('Prediction')

        fig.suptitle('Assignment 3.1\n'
                     'Dataset: {}\n'
                     'Accuracy: {:.3f}'.format(dataset_name.value,
                                               test_accuracy))
        plt.show()
        plt.close()


def main():
    # assignment1()
    # _separator()
    # assignment2()
    # _separator()
    # assignment3()
    # _separator()
    assignment3p1()


if __name__ == '__main__':
    main()
