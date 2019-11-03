import numpy as np


# Short for Multi-Variate Normal.
_mvn = np.random.multivariate_normal


def get_dataset(size, fraction=0.5, seed=0):
    """Returns a dataset of 2D points.

    Args:
        size (int): Number of samples to return.
        fraction (float, optional): Fraction of samples that belongs to the
            positive class. Defaults to 0.5.
        seed (int, optional): Random seed to use. Defaults to 0.

    Returns:
        A numpy.array of samples.
    """
    np.random.seed(seed)

    # The positive samples will be divided into two clusters, so make sure that
    # `num_positive_samples` is even.
    num_positive_samples = int(np.round(fraction * size))
    num_positive_samples += num_positive_samples % 2

    num_negative_samples = size - num_positive_samples

    # Positive cluster #1.
    size_positive_cluster = int(num_positive_samples / 2)
    pos_mean1 = [-1.5, 0.5]
    pos_cov1 = np.array([
        [1., 0.],
        [0., 1.],
    ])
    positive_samples_cluster1 = _mvn(pos_mean1,
                                     pos_cov1,
                                     size=size_positive_cluster)

    # Positive cluster #2.
    pos_mean2 = [1.5, 0.5]
    pos_cov2 = np.array([
        [1., 0.],
        [0., 1.],
    ])
    positive_samples_cluster2 = _mvn(pos_mean2,
                                     pos_cov2,
                                     size=size_positive_cluster)

    positive_samples = np.r_[positive_samples_cluster1,
                             positive_samples_cluster2]

    positive_samples = np.column_stack([positive_samples,
                                        np.ones(num_positive_samples)])

    # Negative cluster.
    neg_mean = [0.0, -0.5]
    neg_cov = np.array([
        [0.5, 0.],
        [0., 0.5],
    ])
    negative_samples = _mvn(neg_mean, neg_cov, size=num_negative_samples)
    negative_samples = np.column_stack([negative_samples,
                                        -np.ones(num_negative_samples)])

    dataset = np.r_[positive_samples, negative_samples]
    np.random.shuffle(dataset)

    return dataset
