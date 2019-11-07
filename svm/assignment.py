import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cvxopt.base import matrix
from cvxopt.solvers import qp

import svm.dataset as ds

sns.set()
_eps = 1e-5

logging.basicConfig(format='[%(levelname)s] %(message)s',
                    level=logging.ERROR)


def _get_indicator_function(alphas, support_vectors, support_labels):
    def _indicator(point):

        # Linear.
        # v1 = np.dot(point[np.newaxis, :], support_vectors.T) + 1

        # Polynomial.
        v1 = np.power(np.dot(point[np.newaxis, :], support_vectors.T) + 1, 1)

        v2 = alphas * support_labels * v1
        res = np.sum(v2)
        return res

    return _indicator


def print_dataset(dataset):
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


def part0():
    dataset = ds.get_dataset(size=20, fraction=0.5)
    # print_dataset(dataset)
    N = len(dataset)

    samples = dataset[:, :-1]
    labels = dataset[:, -1][:, np.newaxis]

    aux_matrix1 = samples @ samples.T
    aux_matrix2 = labels @ labels.T

    # Linear.
    # matrix_P = (aux_matrix1 + 1) * aux_matrix2

    # Polynomial.
    matrix_P = np.power(aux_matrix1 + 1, 1) * aux_matrix2

    np.testing.assert_array_almost_equal(matrix_P, matrix_P.T)

    vector_q = -np.ones(N)
    vector_h = np.zeros(N)
    matrix_G = -np.eye(N)

    res = qp(matrix(matrix_P),
             matrix(vector_q),
             matrix(matrix_G),
             matrix(vector_h))
    alphas = np.array((res['x']))

    nonzero_alphas_indices = np.where(np.abs(alphas) > _eps)[0]
    nonzero_alphas = alphas[nonzero_alphas_indices]
    support_vectors = samples[nonzero_alphas_indices]
    support_labels = labels[nonzero_alphas_indices]

    indic_func = _get_indicator_function(nonzero_alphas,
                                         support_vectors,
                                         support_labels)

    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange(-4, 4, 0.05)
    grid = np.array([
        [
            indic_func(np.array([x, y]))
            for y in yrange
        ]
        for x in xrange
    ])

    pos_samples = dataset[np.where(dataset[:, 2] == 1)]
    neg_samples = dataset[np.where(dataset[:, 2] == -1)]

    assert len(pos_samples) + len(neg_samples) == len(dataset)

    # Plot the dataset.
    plt.figure()

    plt.scatter(pos_samples[:, 0], pos_samples[:, 1],
                color='b', marker='o', label='Positive')
    plt.scatter(neg_samples[:, 0], neg_samples[:, 1],
                color='g', marker='x', label='Negative')

    plt.contour(xrange, yrange, grid)
    # plt.contour(xrange, yrange, grid, levels=(-1, 0, 1))

    plt.gca().set_aspect('equal')
    plt.tight_layout()

    plt.show()
    plt.close()

    print(nonzero_alphas_indices)


def main():
    part0()


if __name__ == '__main__':
    main()
