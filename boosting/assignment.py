import numpy as np
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs

import bayesian_learning.bayesian_classifier as bc

sns.set()


def train_boost(classifier_class,
                samples,
                labels,
                num_iters,
                **classifier_params):
    N, D = samples.shape

    classifiers = []
    alphas = []

    # Initialize the weights uniformly.
    curr_weights = np.ones(N) / N
    for iter_idx in range(num_iters):
        classifier = classifier_class.train(samples=samples,
                                            labels=labels,
                                            weights=curr_weights,
                                            **classifier_params)
        classifiers.append(classifier)

        predictions = classifier.classify(samples)
        mask_correct = predictions == labels
        mask_wrong = ~mask_correct

        weighted_error = np.sum(curr_weights[mask_wrong])
        curr_alpha = \
            0.5 * (np.log(1 - weighted_error) - np.log(weighted_error))
        alphas.append(curr_alpha)

        exponents = curr_alpha * np.ones(N)
        exponents[mask_correct] *= -1
        multipliers = np.exp(exponents)
        new_weights = curr_weights * multipliers
        curr_weights = new_weights / np.sum(new_weights)

        assert np.abs(np.sum(curr_weights) - 1) < 1e-6

    return classifiers, alphas


def classify_boost(samples, classifiers, alphas):
    num_classifiers = len(classifiers)
    if num_classifiers == 1:
        # If there is only one classifier, just use it.
        return classifiers[0].classify(samples)

    N = samples.shape[0]
    classes = classifiers[0].labels
    votes = np.zeros((N, len(classes)))     # (N, C)
    for classifier, alpha in zip(classifiers, alphas):
        assert np.all(classifier.labels == classes)
        curr_predictions = classifier.classify(samples)      # (N,)

        # Use the predictions as indices for the class.
        votes[:, curr_predictions] += alpha

    predictions = np.argmax(votes, axis=1)
    return predictions


def assignment5():
    print('Assignment 5')
    print('Boosted Bayesian classifiers')

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
    classifiers, alphas = train_boost(classifier_class=bc.BayesClassifier,
                                      samples=training_samples,
                                      labels=training_labels,
                                      num_iters=10,
                                      **classifier_params)

    predictions = classify_boost(samples=test_samples,
                                 classifiers=classifiers,
                                 alphas=alphas)

    print(np.mean(predictions == test_labels))


def main():
    assignment5()


if __name__ == '__main__':
    main()
