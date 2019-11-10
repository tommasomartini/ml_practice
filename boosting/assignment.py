import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prettytable import PrettyTable
from sklearn import decomposition
from sklearn.datasets.samples_generator import make_blobs

import bayesian_learning.bayesian_classifier as bayes_cls
import bayesian_learning.dataset as dataset
import bayesian_learning.plotting as plotting
import boosting.boost_classifier as boost_cls

sns.set()


def assignment5():
    print('Assignment 5')
    print('Boosted Bayesian classifiers: first implementation')

    dataset_size = 200
    samples, labels = make_blobs(n_samples=2 * dataset_size,
                                 centers=5,
                                 n_features=2,
                                 random_state=0)
    training_samples = samples[:dataset_size]
    training_labels = labels[:dataset_size]
    test_samples = samples[dataset_size:]
    test_labels = labels[dataset_size:]

    classifier_params = {'naive': False}
    classifier = boost_cls.BoostClassifier.train(
        classifier_class=bayes_cls.BayesClassifier,
        samples=training_samples,
        labels=training_labels,
        num_iters=10,
        **classifier_params)
    predictions = classifier.classify(samples=test_samples)

    print(np.mean(predictions == test_labels))


def assignment5p1():
    num_trials = 10

    pretty_table = PrettyTable()
    pretty_table.field_names = ['Dataset', 'Naive']

    datasets = [
        dataset.DatasetNames.IRIS,
        dataset.DatasetNames.WINE,
        dataset.DatasetNames.VOWEL,
    ]
    for dataset_name in datasets:
        samples, labels = dataset.load_dataset(dataset_name)

        accuracies = []
        for trial_idx in range(num_trials):
            (training_samples,
             training_labels,
             test_samples,
             test_labels) = dataset.split_dataset(samples=samples,
                                                  labels=labels,
                                                  train_fraction=0.5,
                                                  balance_classes=True,
                                                  seed=trial_idx)

            classifier_params = {'naive': False}
            classifier = boost_cls.BoostClassifier.train(
                classifier_class=bayes_cls.BayesClassifier,
                samples=training_samples,
                labels=training_labels,
                num_iters=10,
                **classifier_params)
            predictions = classifier.classify(samples=test_samples)
            test_accuracy = np.mean(predictions == test_labels)
            accuracies.append(test_accuracy)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        def _format_acc(mean_, std_):
            acc_str = '{:.3f} +/- {:.3f}'.format(mean_, std_)
            return acc_str

        pretty_table.add_row([dataset_name.value,
                              _format_acc(mean_accuracy, std_accuracy)])

    print()
    print('Mean accuracy on test set ({} trials)'.format(num_trials))
    print(pretty_table)


def assignment5p2():
    dataset_name = dataset.DatasetNames.IRIS
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

    classifier_params = {'naive': False}
    classifier = boost_cls.BoostClassifier.train(
        classifier_class=bayes_cls.BayesClassifier,
        samples=training_samples,
        labels=training_labels,
        num_iters=10,
        **classifier_params)
    predictions = classifier.classify(samples=test_samples)
    test_accuracy = np.mean(predictions == test_labels)

    # Plot the classification of the test samples.
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plotting.plot_samples_2d(ax=ax1,
                             samples=test_samples,
                             labels=test_labels)
    # plotting.plot_gaussian(ax=ax1,
    #                        samples=test_samples,
    #                        labels=test_labels,
    #                        mu=classifier.mu,
    #                        sigma=classifier.sigma)
    ax1.legend()
    ax1.set_title('Ground truth')

    plotting.plot_samples_2d(ax=ax2,
                             samples=test_samples,
                             labels=predictions)
    # plotting.plot_gaussian(ax=ax2,
    #                        samples=test_samples,
    #                        labels=test_predictions,
    #                        mu=classifier.mu,
    #                        sigma=classifier.sigma)
    plotting.plot_boundaries(ax=ax2,
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
    assignment5p2()


if __name__ == '__main__':
    main()
