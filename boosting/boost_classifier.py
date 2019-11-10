import numpy as np
import seaborn as sns

sns.set()
_eps = 1e-5


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

        weighted_error = np.sum(curr_weights[mask_wrong]) + _eps
        curr_alpha = \
            0.5 * (np.log(1 - weighted_error) - np.log(weighted_error))
        alphas.append(curr_alpha)

        exponents = curr_alpha * np.ones(N)
        exponents[mask_correct] *= -1
        multipliers = np.exp(exponents)
        new_weights = curr_weights * multipliers
        curr_weights = new_weights / np.sum(new_weights)

        try:
            assert np.abs(np.sum(curr_weights) - 1) < 1e-6
        except:
            print()

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
        votes[np.arange(N), curr_predictions] += alpha

    predictions = np.argmax(votes, axis=1)
    return predictions


class BoostClassifier(object):
    def __init__(self, classifiers, alphas):
        self._cassifiers = classifiers
        self._alphas = alphas

        self.labels = []

    @staticmethod
    def train(classifier_class,
              samples,
              labels,
              num_iters,
              **classifier_params):
        classifiers, alphas = train_boost(classifier_class=classifier_class,
                                          samples=samples,
                                          labels=labels,
                                          num_iters=num_iters,
                                          **classifier_params)
        boost_classifier = BoostClassifier(classifiers=classifiers,
                                           alphas=alphas)
        boost_classifier.labels = classifiers[0].labels
        return boost_classifier

    def classify(self, samples):
        predictions = classify_boost(samples=samples,
                                     classifiers=self._cassifiers,
                                     alphas=self._alphas)
        return predictions
