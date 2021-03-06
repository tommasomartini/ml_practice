import copy
import logging

import decision_trees.analysis as analysis

# Maximum depth of the decision tree.
_MAX_DEPTH = 1000000


def max_info_gain_attribute(dataset, attributes):
    """Returns the attribute with the highest average information gain.
    """
    information_gains = [analysis.average_gain(dataset, attribute)
                         for attribute in attributes]

    # A convoluted way to implement argmax.
    split_attributes_idx = max(range(len(information_gains)),
                               key=lambda x: information_gains[x])
    attribute = attributes[split_attributes_idx]
    return attribute


def all_positive(dataset):
    """Returns True if all the samples are positive.
    """
    return all([sample.positive for sample in dataset])


def all_negative(dataset):
    """Returns True if all the samples are positive.
    """
    return not any([sample.positive for sample in dataset])


def compute_accuracy(decision_tree, dataset):
    accuracy = sum([
        decision_tree.classify(sample) == sample.positive
        for sample in dataset
    ]) / len(dataset)
    return accuracy


class Node:

    def __init__(self):
        # Category is the majority class at this point of the tree.
        self.category = None

        # If the node is a leaf, these fields should not be set.
        self.attribute = None
        self.children = {}

    def classify(self, sample):
        if self.is_leaf:
            return self.category

        # Extract the attribute value.
        attribute_value = sample.attribute[self.attribute]

        # Select the child.
        child = self.children[attribute_value]

        # Propagate the classification.
        return child.classify(sample)

    @property
    def size(self):
        if self.is_leaf:
            return 0

        subtree_size = 1 + sum([child.size for child in self.children.values()])
        return subtree_size

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        if self.is_leaf:
            return '+' if self.category else '-'

        string_repr = \
            '{}({})'.format(self.attribute,
                            ''.join(map(str,
                                        [child
                                         for attr_val, child
                                         in sorted(self.children.items())]
                                        )
                                    )
                            )
        return string_repr


def _pruned_subtrees(subtree_root):
    if subtree_root.is_leaf:
        # Nothing to prune.
        return []

    # The trivial way to prune this subtree is by removing all
    # the children from the root.
    leaf_root = Node()
    leaf_root.category = subtree_root.category
    pruned_subtrees = [leaf_root]

    # All the pruned subtrees are given by all the possible ways this
    # node's children can be pruned.
    for attribute_value, child in subtree_root.children.items():
        for alternative_child in _pruned_subtrees(child):
            alternative_root = copy.deepcopy(subtree_root)
            alternative_root.children[attribute_value] = alternative_child
            pruned_subtrees.append(alternative_root)

    return pruned_subtrees


class DecisionTree:

    def __init__(self, root):
        self._root = root

    def classify(self, sample):
        category = self._root.classify(sample)
        return category

    def prune(self, dataset):
        accuracy = compute_accuracy(self, dataset)
        pruned_tree_root = self.root

        logging.debug('  Initial tree: size={:2d}, '
                      'accuracy={:5.1f}, {}'.format(pruned_tree_root.size,
                                                    100 * accuracy,
                                                    pruned_tree_root))

        while True:
            # The candidate tree is the one with highest accuracy and smallest
            # size, if several pruned trees have the same accuracy.
            accuracies_trees = [
                    (compute_accuracy(p_tree, dataset), p_tree)
                    for p_tree in _pruned_subtrees(pruned_tree_root)
                ]
            if len(accuracies_trees) == 0:
                # The current `pruned_tree_root` is made by a single leaf and
                # cannot be pruned.
                break

            candidate_accuracy, candidate_tree = \
                max(accuracies_trees, key=lambda x: (x[0], -x[1].size))

            logging.debug('Candidate tree: size={:2d}, accuracy={:5.1f}, {}'
                          .format(candidate_tree.size,
                                  100 * candidate_accuracy,
                                  candidate_tree))

            if candidate_accuracy < accuracy:
                # All the pruned subtrees perform worse than the previous ones.
                break

            # Found a new best pruned tree.
            accuracy = candidate_accuracy
            pruned_tree_root = candidate_tree

        return pruned_tree_root

    def __repr__(self):
        return str(self._root)

    @property
    def root(self):
        return self._root

    @property
    def size(self):
        return self._root.size

    @staticmethod
    def train(dataset, attributes, max_depth=None):
        if max_depth is None:
            max_depth = _MAX_DEPTH

        if len(dataset) < 1:
            raise ValueError('Empty training set')

        initial_depth = 0
        root = Node()
        stack = [(root, attributes, dataset, initial_depth)]
        while len(stack) > 0:
            node, attributes_left, samples_left, node_depth = stack.pop()

            # Set the default category for this node.
            node.category = analysis.most_common_category(samples_left)

            if node_depth == max_depth:
                # This node cannot have children.
                continue

            # If all the left samples belong to the same category, there is
            # no need to split: make this node a leaf.
            if all_positive(samples_left) or all_negative(samples_left):
                continue

            if not attributes_left:
                # There are no more attributes to split by.
                continue

            # If you get to this point, this node will have children: split
            # using one of the remaining attributes.

            split_attribute = max_info_gain_attribute(
                dataset=samples_left, attributes=attributes_left)
            node.attribute = split_attribute

            # The children nodes will not use this attribute to split.
            # This list could be empty at this point.
            children_attributes = [attr for attr in attributes_left if
                                   attr != split_attribute]

            children_depth = node_depth + 1

            # Create the children.
            for attr_val in split_attribute.values:
                # Create a blank child for this attribute value.
                child = Node()
                node.children[attr_val] = child

                child_samples = analysis.select(dataset=samples_left,
                                                attribute=split_attribute,
                                                value=attr_val)

                if len(child_samples) == 0:
                    # No training samples left. Pick the category with the
                    # biggest representation at the current node.
                    child.category = node.category

                    # Do not put it in the stack.
                    continue

                stack.append((child,
                              children_attributes,
                              child_samples,
                              children_depth))

        decision_tree = DecisionTree(root)
        return decision_tree
