import numpy as np
from prettytable import PrettyTable

import decision_trees.analysis as analysis
import decision_trees.decision_tree as dt
import decision_trees.dataset.monkdata as monkdata

_training_sets = [
    monkdata.monk1,
    monkdata.monk2,
    monkdata.monk3,
]

_testing_sets = [
    monkdata.monk1test,
    monkdata.monk2test,
    monkdata.monk3test,
]


def _separator():
    sep_line = '#' * 80
    print()
    print(sep_line)
    print()


def assignment1():
    print('Assignment 1')
    for dataset_id, dataset in enumerate(_training_sets, start=1):
        dataset_entropy = analysis.entropy(dataset)
        print('  Entropy MONK-{}: {:.3f}'.format(dataset_id, dataset_entropy))


def assignment3():
    print('Assignment 3')
    print('  Information gain for each attribute:')
    pretty_table = PrettyTable()
    pretty_table.field_names = ['Dataset'] + \
                               [attribute for attribute in monkdata.attributes]
    for dataset_id, dataset in enumerate(_training_sets, start=1):
        dataset_name = 'MONK-{}'.format(dataset_id)
        dataset_row = [dataset_name]
        for attribute in monkdata.attributes:
            average_gain = analysis.average_gain(dataset, attribute)
            dataset_row.append('{:.5f}'.format(average_gain))
        pretty_table.add_row(dataset_row)

    print(pretty_table)


def assignment4():
    print('Assignment 4')
    print('Split each dataset based on the attribute with the maximum '
          'average information gain.')

    for dataset_id, dataset in enumerate(_training_sets, start=1):
        dataset_name = 'MONK-{}'.format(dataset_id)
        information_gains = [
            analysis.average_gain(dataset, attribute)
            for attribute in monkdata.attributes
        ]
        split_attribute_idx = np.argmax(information_gains)
        split_attribute = monkdata.attributes[split_attribute_idx]

        print()
        print('Dataset {}'.format(dataset_name))
        print('  Split using attribute {} '
              '(avg info gain {:.5f})'.format(split_attribute,
                                              np.max(information_gains)))
        print('  Entropy for each subset:')

        pretty_table = PrettyTable()
        pretty_table.field_names = \
            ['Full'] + \
            ['{}={}'.format(split_attribute, attribute_value)
             for attribute_value in split_attribute.values]

        subset_entropies = [analysis.entropy(dataset)]
        for attribute_value in split_attribute.values:
            subset = analysis.select(dataset, split_attribute, attribute_value)
            subset_entropy = analysis.entropy(subset)
            subset_entropies.append(subset_entropy)

        pretty_table.add_row(['{:.5f}'.format(v) for v in subset_entropies])
        print(pretty_table)


def assignment4p5():
    print('Assignment 4.5')
    print('Create by hand the first two levels of the decision tree '
          'for the dataset MONK-1.')

    dataset = _training_sets[0]

    information_gains = [
        analysis.average_gain(dataset, attribute)
        for attribute in monkdata.attributes
    ]
    split_attribute_idx = np.argmax(information_gains)
    split_attribute = monkdata.attributes[split_attribute_idx]

    print()
    print('Split using attribute {}.'.format(split_attribute,
                                             np.max(information_gains)))
    pretty_table = PrettyTable()
    field_names = ['All ({})'.format(len(dataset))]
    categories = ['Pos: {}\nNeg: {}'.format(
        len([sample for sample in dataset if sample.positive]),
        len([sample for sample in dataset if not sample.positive]),
    )]
    for subset_idx, attribute_value in enumerate(split_attribute.values,
                                                 start=1):
        subset = analysis.select(dataset, split_attribute, attribute_value)
        field_names.append('Subset {} ({}={}, {})'.format(
            subset_idx,
            split_attribute,
            attribute_value,
            len(subset)))
        categories.append('Pos: {}\nNeg: {}'.format(
        len([sample for sample in subset if sample.positive]),
        len([sample for sample in subset if not sample.positive]),
    ))
    pretty_table.field_names = field_names
    pretty_table.add_row(categories)
    print(pretty_table)

    decision_tree = {
        split_attribute: {attr_val: {} for attr_val in split_attribute.values}
    }

    print()
    print('Subsets:')
    for subset_idx, attribute_value in enumerate(split_attribute.values,
                                                 start=1):
        subset = analysis.select(dataset, split_attribute, attribute_value)
        print(' Subset {} ({}={}, {} elements)'.format(subset_idx,
                                                       split_attribute,
                                                       attribute_value,
                                                       len(subset)))

        subset_information_gains = [analysis.average_gain(subset, attribute)
                                    for attribute in monkdata.attributes]

        print('  Information gain for each attribute:')
        pretty_table = PrettyTable()
        pretty_table.field_names = \
            [attribute for attribute in monkdata.attributes]
        pretty_table.add_row(['{:.5f}'.format(info_gain)
                              for info_gain in subset_information_gains])
        print(pretty_table)

        subset_split_attribute_idx = np.argmax(subset_information_gains)
        subset_split_attribute = monkdata.attributes[subset_split_attribute_idx]

        print('  Split subset {} using attribute {}.'.format(
            subset_idx, subset_split_attribute))

        categories_dict = \
            decision_tree[split_attribute][attribute_value]\
                .setdefault(subset_split_attribute, {})
        for attr_val in subset_split_attribute.values:
            subsubset = analysis.select(subset,
                                        subset_split_attribute,
                                        attr_val)
            subsubset_category = analysis.most_common_category(subsubset)
            categories_dict[attr_val] = subsubset_category

        print()

    # Test my handcrafted decision tree.
    def _predict(sample):
        attr_lvl1 = list(decision_tree.keys())[0]
        value_lvl1 = sample.attribute[attr_lvl1]
        branch_lvl1 = decision_tree[attr_lvl1][value_lvl1]
        attr_lvl2 = list(branch_lvl1.keys())[0]
        value_lvl2 = sample.attribute[attr_lvl2]
        category = branch_lvl1[attr_lvl2][value_lvl2]
        return category

    testset = _testing_sets[0]
    num_matches = sum([
        1 if sample.positive == _predict(sample) else 0
        for sample in testset
    ])
    print('Handcrafted decision tree:')
    print(' {} correct matches out of {}'.format(num_matches, len(testset)))
    print(' Accuracy: {:.5f}'.format(num_matches / len(testset)))


def assignment5():
    print('Assignment 5')
    dataset = _training_sets[0]
    testset = _testing_sets[0]
    decision_tree = dt.build_tree(dataset, monkdata.attributes, maxdepth=2)
    print(decision_tree)
    print(dt.check(decision_tree, testset))
    print(decision_tree.size_subtree)
    # print(d.check(t, m.monk1test))


def assignment5p5():
    print('Assignment 5.5')
    dataset = _training_sets[0]
    testset = _testing_sets[0]
    decision_tree = dt.DecisionTree.train(dataset=dataset,
                                          attributes=monkdata.attributes,
                                          max_depth=2)
    print(decision_tree)


def main():
    # assignment1()
    # _separator()
    # assignment3()
    # _separator()
    # assignment4()
    # _separator()
    # assignment4p5()
    # _separator()
    assignment5()
    _separator()
    assignment5p5()


if __name__ == '__main__':
    main()
