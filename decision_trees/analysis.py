import math


def log2(x):
    return math.log(x, 2)


def entropy(dataset):
    n = len(dataset)
    n_pos = len([x for x in dataset if x.positive])
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    p_pos = n_pos / n
    p_neg = n_neg / n

    entropy_val = - p_pos * log2(p_pos) - p_neg * log2(p_neg)
    return entropy_val


def select(dataset, attribute, value):
    """Returns the subset of data samples where the attribute has
    the given value.
    """
    subset = [x for x in dataset if x.attribute[attribute] == value]
    return subset


def average_gain(dataset, attribute):
    dataset_size = len(dataset)
    partitioned_entropy = 0.0
    for v in attribute.values:
        subset = select(dataset, attribute, v)
        subset_entropy = entropy(subset)
        weight = len(subset) / dataset_size
        partitioned_entropy += weight * subset_entropy

    avg_gain = entropy(dataset) - partitioned_entropy
    return avg_gain


def most_common_category(dataset):
    """Returns the majority binary class (True or False) for this dataset.
    """
    pos_count = len([sample for sample in dataset if sample.positive])
    neg_count = len(dataset) - pos_count
    return pos_count > neg_count
