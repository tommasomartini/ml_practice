import unittest

import bayesian_learning.bayesian_classifier as bc
import bayesian_learning.dataset as dataset


class RegressionTests(unittest.TestCase):

    def test_accuracy_regression(self):
        expected_accuracies = {
            dataset.DatasetNames.IRIS: 0.96,
            dataset.DatasetNames.WINE: 0.9772727272727273,
            dataset.DatasetNames.VOWEL: 0.8560606060606061,
        }

        for dataset_name, expected_accuracy in expected_accuracies.items():
            samples, labels = dataset.load_dataset(dataset_name)

            (training_samples,
             training_labels,
             test_samples,
             test_labels) = dataset.split_dataset(samples=samples,
                                                  labels=labels,
                                                  train_fraction=0.5,
                                                  balance_classes=True,
                                                  seed=0)

            classifier = bc.BayesClassifier.train(samples=training_samples,
                                                  labels=training_labels,
                                                  naive=False)
            test_predictions = classifier.classify(samples=test_samples)
            test_accuracy = bc.evaluate_accuracy(
                predictions=test_predictions,
                labels=test_labels)

            self.assertAlmostEqual(expected_accuracy, test_accuracy, places=6)


if __name__ == '__main__':
    unittest.main()
