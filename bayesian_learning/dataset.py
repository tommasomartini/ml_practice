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