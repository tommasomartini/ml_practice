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


# class DecisionTree:
#
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def train(dataset, max_depth=None):
#         max_depth = max_depth or _MAX_DEPTH
#         tree = DecisionTree()
#         return tree
