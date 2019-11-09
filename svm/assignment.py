import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cvxopt.base import matrix
from cvxopt.solvers import qp

import svm.dataset as ds
import svm.kernels as kernels
import svm.plotting as plotting

sns.set()
_eps = 1e-5
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.ERROR)


_kernel_func = kernels.get_linear_kernel()
_training_set = ds.get_dataset(size=20, fraction=0.5, seed=0)
_test_set = ds.get_dataset(size=20, fraction=0.5, seed=1)


def _get_indicator_function(support_vectors,
                            support_labels,
                            alphas,
                            kernel_func):
    """Returns an indicator function to classify points.

    Args:
        support_vectors: Shape=(K, D) with K number of support vectors and D
            their dimensionality.
        support_labels: Shape=(K,) labels for each of the K support vectors.
        alphas: Shape=(K,) non-zero coefficients for each of
            the K support vectors.
        kernel_func: Kernel function (See the kernels module for
            its function signature).

    Returns:
        An indicator function.
    """
    def _indicator(*features):
        """The classification of the input samples is given by the sign of the
        returned value.

        Args:
            *features: Each feature can be either a scalar or an array-like
                with shape=(N,) containing the corresponding feature for each
                of the N samples.

        Returns:
            A scalar or an array-like with shape=(N,). The sample
            classifications corresponds to the sign of the returned value.
        """
        points = np.column_stack(features)  # (N, D)

        v1 = kernel_func(points, support_vectors)   # (N, K)

        K = len(support_labels)
        alphas_r = alphas.reshape((-1, K))
        support_labels_r = support_labels.reshape((-1, K))

        v2 = alphas_r * support_labels_r * v1   # (N, K)
        res = np.sum(v2, axis=1)    # (N,)
        return res

    return _indicator


def _evaluate_model(indicator_function, dataset):
    samples = dataset[:, :-1]
    labels = dataset[:, -1]
    predictions = np.sign(indicator_function(samples))
    accuracy = np.mean(labels == predictions)
    print('Accuracy: {:.3f}'.format(accuracy))


def _plot(dataset, indicator_function, title):
    plt.figure()
    ax = plt.gca()
    plotting.plot_data_points_and_margin(
        ax=ax,
        dataset=dataset,
        indicator_function=indicator_function,
        grid_size=None)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()


def main():
    N = len(_training_set)
    samples = _training_set[:, :-1]
    labels = _training_set[:, -1][:, np.newaxis]

    matrix_P = _kernel_func(samples, samples) * (labels @ labels.T)
    vector_q = -np.ones(N)
    vector_h = np.zeros(N)
    matrix_G = -np.eye(N)

    np.testing.assert_array_almost_equal(matrix_P, matrix_P.T)

    res = qp(matrix(matrix_P),
             matrix(vector_q),
             matrix(matrix_G),
             matrix(vector_h))

    if res['status'] != 'optimal':
        raise ValueError('Could not solve the problem')

    alphas = np.squeeze(res['x'])
    nonzero_alphas_indices = np.where(np.abs(alphas) > _eps)[0]
    nonzero_alphas = alphas[nonzero_alphas_indices]
    support_vectors = samples[nonzero_alphas_indices]
    support_labels = np.squeeze(labels[nonzero_alphas_indices])

    indic_func = _get_indicator_function(support_vectors=support_vectors,
                                         support_labels=support_labels,
                                         alphas=nonzero_alphas,
                                         kernel_func=_kernel_func)

    _evaluate_model(indicator_function=indic_func,
                    dataset=_test_set)

    _plot(dataset=_training_set,
          indicator_function=indic_func,
          title='Training samples')
    _plot(dataset=_test_set,
          indicator_function=indic_func,
          title='Test samples')


if __name__ == '__main__':
    main()
