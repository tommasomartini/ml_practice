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
import boosting.decision_tree_classifier as tree_cls

sns.set()


def _separator():
    sep_line = '#' * 80
    print()
    print(sep_line)
    print()


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
    print('Assignment 5.1')
    print('Compare weak classifiers and boosted classifiers on real datasets.')

    num_trials = 10

    pretty_table = PrettyTable()
    pretty_table.field_names = ['Dataset', 'Weak', 'Boosted']

    for dataset_name in dataset.DatasetNames:
        samples, labels = dataset.load_dataset(dataset_name)

        if dataset_name == dataset.DatasetNames.OLIVETTI:
            # The dimensionality of the Olivetti dataset is too big. Use PCA
            # to make the problem tractable.
            pca = decomposition.PCA(n_components=20)
            pca.fit(samples)
            samples = pca.transform(samples)

        weak_accuracies = []
        boosted_accuracies = []
        for trial_idx in range(num_trials):
            (training_samples,
             training_labels,
             test_samples,
             test_labels) = dataset.split_dataset(samples=samples,
                                                  labels=labels,
                                                  train_fraction=0.7,
                                                  balance_classes=True,
                                                  seed=trial_idx)

            weak_classifier = \
                bayes_cls.BayesClassifier.train(samples=training_samples,
                                                labels=training_labels,
                                                naive=True,
                                                weights=None)
            weak_predictions = weak_classifier.classify(samples=test_samples)
            weak_accuracy = np.mean(weak_predictions == test_labels)
            weak_accuracies.append(weak_accuracy)

            classifier_params = {'naive': True}
            boost_classifier = boost_cls.BoostClassifier.train(
                classifier_class=bayes_cls.BayesClassifier,
                samples=training_samples,
                labels=training_labels,
                num_iters=10,
                **classifier_params)
            boost_predictions = boost_classifier.classify(samples=test_samples)
            boost_accuracy = np.mean(boost_predictions == test_labels)
            boosted_accuracies.append(boost_accuracy)

        mean_weak_accuracy = np.mean(weak_accuracies)
        std_weak_accuracy = np.std(weak_accuracies)
        mean_boosted_accuracy = np.mean(boosted_accuracies)
        std_boosted_accuracy = np.std(boosted_accuracies)

        def _format_acc(mean_, std_):
            acc_str = '{:.3f} +/- {:.3f}'.format(mean_, std_)
            return acc_str

        pretty_table.add_row([dataset_name.value,
                              _format_acc(mean_weak_accuracy,
                                          std_weak_accuracy),
                              _format_acc(mean_boosted_accuracy,
                                          std_boosted_accuracy)])

    print()
    print('Mean accuracy on test set ({} trials)'.format(num_trials))
    print(pretty_table)


def assignment5p2():
    print('Assignment 5.2')
    print('Compare the boundaries of weak and boosted classifiers.')

    for dataset_name in dataset.DatasetNames:
        samples, labels = dataset.load_dataset(dataset_name)
        pca = decomposition.PCA(n_components=2)
        pca.fit(samples)
        samples = pca.transform(samples)

        (training_samples,
         training_labels,
         test_samples,
         test_labels) = dataset.split_dataset(samples=samples,
                                              labels=labels,
                                              train_fraction=0.7,
                                              balance_classes=True,
                                              seed=0)

        weak_classifier = \
            bayes_cls.BayesClassifier.train(samples=training_samples,
                                            labels=training_labels,
                                            naive=True,
                                            weights=None)
        weak_predictions = weak_classifier.classify(samples=test_samples)
        weak_accuracy = np.mean(weak_predictions == test_labels)

        classifier_params = {'naive': True}
        boost_classifier = boost_cls.BoostClassifier.train(
            classifier_class=bayes_cls.BayesClassifier,
            samples=training_samples,
            labels=training_labels,
            num_iters=10,
            **classifier_params)
        boost_predictions = boost_classifier.classify(samples=test_samples)
        boost_accuracy = np.mean(boost_predictions == test_labels)

        # Plot the classification of the test samples.
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plotting.plot_samples_2d(ax=ax1,
                                 samples=test_samples,
                                 labels=test_labels)
        ax1.legend()
        ax1.set_title('Ground truth')

        plotting.plot_samples_2d(ax=ax2,
                                 samples=test_samples,
                                 labels=weak_predictions)
        plotting.plot_boundaries(ax=ax2,
                                 classifier=weak_classifier,
                                 grid_size=1000)
        ax2.legend()
        ax2.set_title('Weak classifier')

        plotting.plot_samples_2d(ax=ax3,
                                 samples=test_samples,
                                 labels=boost_predictions)
        plotting.plot_boundaries(ax=ax3,
                                 classifier=boost_classifier,
                                 grid_size=1000)
        ax3.legend()
        ax3.set_title('Boosted classifier')

        fig.suptitle('Assignment 5.2\n'
                     'Dataset: {}\n'
                     'Weak classifier accuracy: {:.3f}\n'
                     'Boosted classifier accuracy: {:.3f}'.format(
            dataset_name.value, weak_accuracy, boost_accuracy))

        plt.show()
        plt.close()


def assignment6():
    print('Assignment 6')
    print('Boosted trees, using Sklearn implementation.')

    num_trials = 10

    pretty_table = PrettyTable()
    pretty_table.field_names = ['Dataset', 'Weak', 'Boosted']

    for dataset_name in dataset.DatasetNames:
        samples, labels = dataset.load_dataset(dataset_name)

        if dataset_name == dataset.DatasetNames.OLIVETTI:
            # The dimensionality of the Olivetti dataset is too big. Use PCA
            # to make the problem tractable.
            pca = decomposition.PCA(n_components=20)
            pca.fit(samples)
            samples = pca.transform(samples)

        weak_accuracies = []
        boosted_accuracies = []
        for trial_idx in range(num_trials):
            (training_samples,
             training_labels,
             test_samples,
             test_labels) = dataset.split_dataset(samples=samples,
                                                  labels=labels,
                                                  train_fraction=0.7,
                                                  balance_classes=True,
                                                  seed=trial_idx)

            weak_classifier = \
                tree_cls.SklearnDecisionTreeClassifierWrapper.train(
                    samples=training_samples,
                    labels=training_labels,
                    weights=None)
            weak_predictions = weak_classifier.classify(samples=test_samples)
            weak_accuracy = np.mean(weak_predictions == test_labels)
            weak_accuracies.append(weak_accuracy)

            boost_classifier = boost_cls.BoostClassifier.train(
                classifier_class=tree_cls.SklearnDecisionTreeClassifierWrapper,
                samples=training_samples,
                labels=training_labels,
                num_iters=10)
            boost_predictions = boost_classifier.classify(samples=test_samples)
            boost_accuracy = np.mean(boost_predictions == test_labels)
            boosted_accuracies.append(boost_accuracy)

        mean_weak_accuracy = np.mean(weak_accuracies)
        std_weak_accuracy = np.std(weak_accuracies)
        mean_boosted_accuracy = np.mean(boosted_accuracies)
        std_boosted_accuracy = np.std(boosted_accuracies)

        def _format_acc(mean_, std_):
            acc_str = '{:.3f} +/- {:.3f}'.format(mean_, std_)
            return acc_str

        pretty_table.add_row([dataset_name.value,
                              _format_acc(mean_weak_accuracy,
                                          std_weak_accuracy),
                              _format_acc(mean_boosted_accuracy,
                                          std_boosted_accuracy)])

    print()
    print('Mean accuracy on test set ({} trials)'.format(num_trials))
    print(pretty_table)


def assignment6p1():
    print('Assignment 6.1')
    print('Boosted trees, using Sklearn implementation: '
          'plot the boundaries.')

    for dataset_name in dataset.DatasetNames:
        samples, labels = dataset.load_dataset(dataset_name)
        pca = decomposition.PCA(n_components=2)
        pca.fit(samples)
        samples = pca.transform(samples)

        (training_samples,
         training_labels,
         test_samples,
         test_labels) = dataset.split_dataset(samples=samples,
                                              labels=labels,
                                              train_fraction=0.7,
                                              balance_classes=True,
                                              seed=0)

        weak_classifier = \
            tree_cls.SklearnDecisionTreeClassifierWrapper.train(
                samples=training_samples,
                labels=training_labels,
                weights=None)
        weak_predictions = weak_classifier.classify(samples=test_samples)
        weak_accuracy = np.mean(weak_predictions == test_labels)

        boost_classifier = boost_cls.BoostClassifier.train(
            classifier_class=tree_cls.SklearnDecisionTreeClassifierWrapper,
            samples=training_samples,
            labels=training_labels,
            num_iters=10)
        boost_predictions = boost_classifier.classify(samples=test_samples)
        boost_accuracy = np.mean(boost_predictions == test_labels)

        # Plot the classification of the test samples.
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plotting.plot_samples_2d(ax=ax1,
                                 samples=test_samples,
                                 labels=test_labels)
        ax1.legend()
        ax1.set_title('Ground truth')

        plotting.plot_samples_2d(ax=ax2,
                                 samples=test_samples,
                                 labels=weak_predictions)
        plotting.plot_boundaries(ax=ax2,
                                 classifier=weak_classifier,
                                 grid_size=1000)
        ax2.legend()
        ax2.set_title('Weak classifier')

        plotting.plot_samples_2d(ax=ax3,
                                 samples=test_samples,
                                 labels=boost_predictions)
        plotting.plot_boundaries(ax=ax3,
                                 classifier=boost_classifier,
                                 grid_size=1000)
        ax3.legend()
        ax3.set_title('Boosted classifier')

        fig.suptitle('Assignment 5.2\n'
                     'Dataset: {}\n'
                     'Weak classifier accuracy: {:.3f}\n'
                     'Boosted classifier accuracy: {:.3f}'.format(
            dataset_name.value, weak_accuracy, boost_accuracy))

        plt.show()
        plt.close()


def main():
    # assignment5()
    # _separator()
    # assignment5p1()
    # _separator()
    # assignment5p2()
    # _separator()
    # assignment6()
    # _separator()
    assignment6p1()


if __name__ == '__main__':
    main()
