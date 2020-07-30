import unittest

import numpy as np

import common.metrics as metrics


class TestPrecision(unittest.TestCase):
    def test_success(self):
        label = np.array([0, 1, 1, 0])
        prediction = np.array([0, 0, 1, 1])
        expected_precision = 0.5

        precision = metrics.compute_precision(label, prediction)

        self.assertAlmostEqual(expected_precision, precision)

    def test_100_precision(self):
        label = np.array([0, 1, 1, 0])
        prediction = np.array([0, 1, 0, 0])
        expected_precision = 1.0

        precision = metrics.compute_precision(label, prediction)

        self.assertAlmostEqual(expected_precision, precision)

    def test_no_positive_prediction(self):
        label = np.array([0, 1, 1, 0])
        prediction = np.array([0, 0, 0, 0])
        expected_precision = 0.0

        precision = metrics.compute_precision(label, prediction)

        self.assertAlmostEqual(expected_precision, precision)


class TestRecall(unittest.TestCase):
    def test_success(self):
        label = np.array([0, 1, 1, 0])
        prediction = np.array([0, 0, 1, 1])
        expected_recall = 0.5

        recall = metrics.compute_recall(label, prediction)

        self.assertAlmostEqual(expected_recall, recall)

    def test_100_recall(self):
        label = np.array([0, 1, 1, 0])
        prediction = np.array([1, 1, 1, 1])
        expected_recall = 1.0

        recall = metrics.compute_recall(label, prediction)

        self.assertAlmostEqual(expected_recall, recall)

    def test_no_positive_label(self):
        label = np.array([0, 0, 0, 0])
        prediction = np.array([0, 1, 1, 0])
        expected_recall = 1.0

        recall = metrics.compute_recall(label, prediction)

        self.assertAlmostEqual(expected_recall, recall)


if __name__ == '__main__':
    unittest.main()
