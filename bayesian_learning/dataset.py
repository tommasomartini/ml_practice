import enum
import os
from pathlib import Path

import numpy as np
from numpy import genfromtxt

_this_file_dir = str(Path(__file__).parent.absolute())
_DATA_DIR = os.path.join(_this_file_dir, 'data')


class DatasetNames(enum.Enum):
    IRIS = 'iris'
    WINE = 'wine'
    OLIVETTI = 'olivetti'
    VOWEL = 'vowel'


# Maps each dataset to a tuple of paths (samples, labels).
_dataset_paths = {
    DatasetNames.IRIS: (
        os.path.join(_DATA_DIR, 'irisX.txt'),
        os.path.join(_DATA_DIR, 'irisY.txt'),
    ),
    DatasetNames.WINE: (
        os.path.join(_DATA_DIR, 'wineX.txt'),
        os.path.join(_DATA_DIR, 'wineY.txt'),
    ),
    DatasetNames.OLIVETTI: (
        os.path.join(_DATA_DIR, 'olivettifacesX.txt'),
        os.path.join(_DATA_DIR, 'olivettifacesY.txt'),
    ),
    DatasetNames.VOWEL: (
        os.path.join(_DATA_DIR, 'vowelX.txt'),
        os.path.join(_DATA_DIR, 'vowelY.txt'),
    ),
}


def load_dataset(dataset_name):

    if dataset_name == DatasetNames.IRIS:
        samples_path, labels_path = _dataset_paths[DatasetNames.IRIS]
        samples = genfromtxt(samples_path, delimiter=',')
        labels = genfromtxt(labels_path, delimiter=',', dtype=np.int) - 1
        pcadim = 2

    elif dataset_name == DatasetNames.WINE:
        samples_path, labels_path = _dataset_paths[DatasetNames.WINE]
        samples = genfromtxt(samples_path, delimiter=',')
        labels = genfromtxt(labels_path, delimiter=',', dtype=np.int) - 1
        pcadim = 0

    elif dataset_name == DatasetNames.OLIVETTI:
        samples_path, labels_path = _dataset_paths[DatasetNames.OLIVETTI]
        samples = genfromtxt(samples_path, delimiter=',')
        samples = samples / 255
        labels = genfromtxt(labels_path, delimiter=',', dtype=np.int)
        pcadim = 20

    elif dataset_name == DatasetNames.VOWEL:
        samples_path, labels_path = _dataset_paths[DatasetNames.VOWEL]
        samples = genfromtxt(samples_path, delimiter=',')
        labels = genfromtxt(labels_path, delimiter=',', dtype=np.int)
        pcadim = 0

    else:
        raise ValueError('Unknown dataset {}'.format(dataset_name))

    return samples, labels, pcadim


def split_dataset(samples, labels, train_fraction=0.5, seed=None):
    N, _D = samples.shape
    assert labels.shape == (N,)

    training_size = int(np.rint(N * train_fraction))

    np.random.seed(seed)
    shuffled_indices = np.random.permutation(N)
    training_indices = shuffled_indices[:training_size]
    test_indices = shuffled_indices[training_size:]

    training_samples = samples[training_indices, :]
    training_labels = labels[training_indices]

    test_samples = samples[test_indices, :]
    test_labels = labels[test_indices]

    return training_samples, training_labels, test_samples, test_labels, training_indices, test_indices
