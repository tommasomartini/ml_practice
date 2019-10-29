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


class TreeNode:
    """Decision tree representation.
    """

    def __init__(self, attribute, branches, default):
        self.attribute = attribute
        self.branches = branches
        self.default = default

    def __repr__(self):
        # repr_str = str(self.attribute) \
        #            + '({})'.format()
        accum = str(self.attribute) + '('
        for x in sorted(self.branches):
            accum += str(self.branches[x])
        return accum + ')'

    @property
    def size_subtree(self):
        accum = 1
        for x in sorted(self.branches):
            accum += self.branches[x].size_subtree
        return accum


class TreeLeaf:
    """Decision tree representation for leaf nodes.

    Leaf nodes only encode the category assigned to that end of the node.
    """

    def __init__(self, category):
        self.category = category

    def __repr__(self):
        return '+' if self.category else '-'

    @property
    def size_subtree(self):
        return 0


def build_tree(dataset, attributes, maxdepth=10000):
    """Recursively builds a decision tree trained on a dataset.
    """

    def buildBranch(dataset, default, attributes):
        if not dataset:
            return TreeLeaf(default)

        if all_positive(dataset):
            return TreeLeaf(True)

        if all_negative(dataset):
            return TreeLeaf(False)

        return build_tree(dataset, attributes, maxdepth - 1)

    # The default category is the one with most samples in the dataset.
    default_category = analysis.most_common_category(dataset)
    if maxdepth < 1:
        # Build a tree always predicting the most common class in the dataset.
        return TreeLeaf(default_category)

    split_attribute = max_info_gain_attribute(dataset, attributes)
    attributes_left = [attr for attr in attributes if attr != split_attribute]
    branches = [
        (v, buildBranch(analysis.select(dataset, split_attribute, v), default_category, attributes_left))
        for v in split_attribute.values
    ]
    root = TreeNode(split_attribute, dict(branches), default_category)
    return root


def classify(tree, sample):
    "Classify a sample using the given decition tree"
    if isinstance(tree, TreeLeaf):
        return tree.category
    return classify(tree.branches[sample.attribute[tree.attribute]], sample)


def check(tree, testdata):
    "Measure fraction of correctly classified samples"
    correct = 0
    for x in testdata:
        if classify(tree, x) == x.positive:
            correct += 1
    return float(correct)/len(testdata)


def compute_accuracy(decision_tree, dataset):
    accuracy = sum([
        decision_tree.classify(sample) == sample.positive
        for sample in dataset
    ]) / len(dataset)
    return accuracy


class Node:

    def __init__(self):
        # If the node is a leaf, this is the only field to set.
        self.category = None

        # If the node is not a leaf, these are the only fields to set.
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
    def is_leaf(self):
        return self.category is not None

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

            if node_depth == max_depth:
                # This node cannot have children: make it a leaf by assigning
                # the most common category.
                node.category = analysis.most_common_category(samples_left)
                continue

            # If all the left samples belong to the same category, there is
            # no need to split: make this node a leaf.
            if all_positive(samples_left):
                node.category = True
                continue

            if all_negative(samples_left):
                node.category = False
                continue

            if not attributes_left:
                # There are no more attributes to split by. Make this node
                # a leaf by using the most common category.
                node.category = analysis.most_common_category(samples_left)
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
                    child.category = analysis.most_common_category(samples_left)

                    # Do not put it in the stack.
                    continue

                stack.append((child,
                              children_attributes,
                              child_samples,
                              children_depth))

        decision_tree = DecisionTree(root)
        return decision_tree
