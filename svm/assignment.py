import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import svm.dataset as ds

sns.set()

logging.basicConfig(format='[%(levelname)s] %(message)s',
                    level=logging.ERROR)


def part1():
    dataset = ds.get_dataset(size=200, fraction=0.5)

    pos_samples = dataset[np.where(dataset[:, 2] == 1)]
    neg_samples = dataset[np.where(dataset[:, 2] == -1)]

    assert len(pos_samples) + len(neg_samples) == len(dataset)

    # Plot the dataset.
    plt.figure()

    plt.scatter(pos_samples[:, 0], pos_samples[:, 1],
               color='b', marker='o', label='Positive')
    plt.scatter(neg_samples[:, 0], neg_samples[:, 1],
                color='g', marker='x', label='Negative')

    plt.gca().set_aspect('equal')
    plt.tight_layout()

    plt.show()
    plt.close()


def main():
    part1()


if __name__ == '__main__':
    main()
