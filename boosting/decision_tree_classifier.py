from sklearn import tree
import numpy as np


class SklearnDecisionTreeClassifierWrapper:

    def __init__(self, sklearn_tree):
        self._sklearn_tree = sklearn_tree

        self.labels = []

    @staticmethod
    def train(samples, labels, weights=None):
        max_depth = samples.shape[1] / 2 + 1
        sklearn_tree = tree.DecisionTreeClassifier(max_depth=max_depth)
        if weights is not None:
            sklearn_tree.fit(samples, labels, sample_weight=weights.flatten())
        else:
            sklearn_tree.fit(samples, labels)

        decision_tree_wrapper = \
            SklearnDecisionTreeClassifierWrapper(sklearn_tree)

        decision_tree_wrapper.labels = np.unique(labels)

        return  decision_tree_wrapper

    def classify(self, samples):
        predictions = self._sklearn_tree.predict(samples)
        return predictions
