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

    elif dataset_name == DatasetNames.WINE:
        samples_path, labels_path = _dataset_paths[DatasetNames.WINE]
        samples = genfromtxt(samples_path, delimiter=',')
        labels = genfromtxt(labels_path, delimiter=',', dtype=np.int) - 1

    elif dataset_name == DatasetNames.OLIVETTI:
        samples_path, labels_path = _dataset_paths[DatasetNames.OLIVETTI]
        samples = genfromtxt(samples_path, delimiter=',')
        samples = samples / 255
        labels = genfromtxt(labels_path, delimiter=',', dtype=np.int)

    elif dataset_name == DatasetNames.VOWEL:
        samples_path, labels_path = _dataset_paths[DatasetNames.VOWEL]
        samples = genfromtxt(samples_path, delimiter=',')
        labels = genfromtxt(labels_path, delimiter=',', dtype=np.int)

    else:
        raise ValueError('Unknown dataset {}'.format(dataset_name))

    return samples, labels


def _random_split(elements, fraction):
    size = len(elements)
    split1_size = int(np.rint(size * fraction))
    shuffled_elements = np.random.permutation(elements)
    split1 = shuffled_elements[:split1_size]
    split2 = shuffled_elements[split1_size:]
    return split1, split2


def split_dataset(samples,
                  labels,
                  train_fraction=0.5,
                  balance_classes=False,
                  seed=None):
    N, _D = samples.shape
    assert labels.shape == (N,)

    np.random.seed(seed)

    if not balance_classes:
        training_indices, test_indices = _random_split(elements=np.arange(N),
                                                       fraction=train_fraction)
    else:
        training_indices = np.array([], dtype=int)
        test_indices = np.array([], dtype=int)
        for idx, class_id in enumerate(np.unique(labels)):
            class_training_indices, class_test_indices = \
                _random_split(elements=np.where(labels == class_id)[0],
                              fraction=train_fraction)
            training_indices = np.concatenate((training_indices,
                                               class_training_indices))
            test_indices = np.concatenate((test_indices, class_test_indices))

    training_samples = samples[training_indices, :]
    training_labels = labels[training_indices]

    test_samples = samples[test_indices, :]
    test_labels = labels[test_indices]

    return (training_samples,
            training_labels,
            test_samples,
            test_labels)
