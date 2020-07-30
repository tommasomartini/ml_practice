import numpy as np


def compute_accuracy(label, prediction):
    correct_predictions = label == prediction
    accuracy = np.mean(correct_predictions)
    return accuracy


def compute_precision(label, prediction):
    """Computes:
        TP / (TP + FP)

    How many of the samples I classified as positives are actually positives?
    """
    # Only consider the positive predictions.
    positive_prediction_index = prediction == 1
    prediction = prediction[positive_prediction_index]
    label = label[positive_prediction_index]

    tp = sum(label == prediction)
    fp = len(prediction) - tp
    if tp + fp == 0:
        return 0

    precision = tp / (tp + fp)
    return precision


def compute_recall(label, prediction):
    """Computes:
        TP / (TP + FN)

    How many of the positive samples did I detect?
    """
    # Only consider the positive labels.
    positive_label_index = label == 1
    prediction = prediction[positive_label_index]
    label = label[positive_label_index]

    tp = sum(label == prediction)
    fn = len(prediction) - tp

    if tp + fn == 0:
        return 1.0

    recall = tp / (tp + fn)
    return recall
