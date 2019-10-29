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


class DecisionTree:

    def __init__(self, root):
        self._root = root

    def classify(self, sample):
        category = self._root.classify(sample)
        return category

    def __repr__(self):
        return str(self._root)

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
