import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prettytable import PrettyTable
from tqdm import tqdm

import decision_trees.analysis as analysis
import decision_trees.dataset.monkdata as monkdata
import decision_trees.decision_tree as dt
import decision_trees.provided.dtree as provided

logging.basicConfig(format='[%(levelname)s] %(message)s',
                    level=logging.ERROR)

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
            decision_tree[split_attribute][attribute_value] \
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
    print('Provided decision tree implementation')

    pretty_table = PrettyTable()
    pretty_table.field_names = ['Dataset',
                                'Accuracy training set (%)',
                                'Accuracy test set (%)',
                                'Size']
    for dataset_id, (training_set, testset) in enumerate(zip(_training_sets,
                                                             _testing_sets),
                                                         start=1):
        dataset_name = 'MONK-{}'.format(dataset_id)
        decision_tree = provided.buildTree(training_set, monkdata.attributes)
        train_acc = provided.check(decision_tree, training_set)
        test_acc = provided.check(decision_tree, testset)
        pretty_table.add_row([dataset_name,
                              '{:.1f}'.format(100 * train_acc),
                              '{:.1f}'.format(100 * test_acc),
                              decision_tree.size_subtree])
    print(pretty_table)


def assignment5p5():
    print('Assignment 5.5')
    print('My decision tree implementation')

    pretty_table = PrettyTable()
    pretty_table.field_names = ['Dataset',
                                'Accuracy training set (%)',
                                'Accuracy test set (%)',
                                'Size']
    for dataset_id, (training_set, testset) in enumerate(zip(_training_sets,
                                                             _testing_sets),
                                                         start=1):
        dataset_name = 'MONK-{}'.format(dataset_id)
        decision_tree = dt.DecisionTree.train(dataset=training_set,
                                              attributes=monkdata.attributes)
        train_acc = dt.compute_accuracy(decision_tree, training_set)
        test_acc = dt.compute_accuracy(decision_tree, testset)
        pretty_table.add_row([dataset_name,
                              '{:.1f}'.format(100 * train_acc),
                              '{:.1f}'.format(100 * test_acc),
                              decision_tree.size])
    print(pretty_table)


def assignment7p0():
    print('Assignment 7.0')
    print('Use the provided routines to prune the tree.')

    testset = _testing_sets[0]
    training_set = _training_sets[0]
    training_set, validation_set = monkdata.partition(training_set,
                                                      fraction=0.5,
                                                      seed=0)
    decision_tree = provided.buildTree(training_set, monkdata.attributes)

    print()

    print('Full tree')
    print(' Size: {}'.format(decision_tree.size_subtree))
    print(' Validation set: '
          '{:.1f}%'.format(100 * provided.check(decision_tree, validation_set)))
    print(' Test set: '
          '{:.1f}%'.format(100 * provided.check(decision_tree, testset)))

    print()

    pruned_accuracy = provided.check(decision_tree, validation_set)
    pruned_tree = decision_tree

    logging.debug('  Initial tree: size={:2d}, '
                  'accuracy={:5.1f}, {}'.format(pruned_tree.size_subtree,
                                                100 * pruned_accuracy,
                                                pruned_tree))

    while True:
        candidate_accuracy, candidate_tree = max(
            [
                (provided.check(p_tree, validation_set), p_tree)
                for p_tree in provided.allPruned(pruned_tree)
            ],
            key=lambda x: (x[0], -x[1].size_subtree)
        )

        logging.debug('Candidate tree: size={:2d}, accuracy={:5.1f}, {}'
                      .format(candidate_tree.size_subtree,
                              100 * candidate_accuracy,
                              candidate_tree))

        if candidate_accuracy < pruned_accuracy:
            # All the pruned subtree perform worse than the previous ones.
            break

        # Found a new best pruned tree.
        pruned_accuracy = candidate_accuracy
        pruned_tree = candidate_tree

    print('Pruned tree')
    print(' Size: {}'.format(pruned_tree.size_subtree))
    print(' Validation set: '
          '{:.1f}%'.format(100 * provided.check(pruned_tree, validation_set)))
    print(' Test set: '
          '{:.1f}%'.format(100 * provided.check(pruned_tree, testset)))


def assignment7p1():
    print('Assignment 7.1')
    print('Use my implementation to prune the tree.')

    testset = _testing_sets[0]
    training_set = _training_sets[0]
    training_set, validation_set = monkdata.partition(training_set,
                                                      fraction=0.5,
                                                      seed=0)
    decision_tree = dt.DecisionTree.train(training_set, monkdata.attributes)

    print()

    print('Full tree')
    print(' Size: {}'.format(decision_tree.size))
    print(' Validation set: '
          '{:.1f}%'.format(100 * dt.compute_accuracy(decision_tree,
                                                     validation_set)))
    print(' Test set: '
          '{:.1f}%'.format(100 * dt.compute_accuracy(decision_tree, testset)))
    print()

    pruned_tree = decision_tree.prune(validation_set)

    print('Pruned tree')
    print(' Size: {}'.format(pruned_tree.size))
    print(' Validation set: '
          '{:.1f}%'.format(100 * dt.compute_accuracy(pruned_tree,
                                                     validation_set)))
    print(' Test set: '
          '{:.1f}%'.format(100 * dt.compute_accuracy(pruned_tree, testset)))


def assignment7p2():
    print('Assignment 7.2')
    print('Run my implementation to prune the tree on all the datasets.')
    print()

    # How many times to repeat each experiment.
    num_repetitions = 100

    # All the train vs validation splits to test.
    train_val_fractions = np.linspace(0.1, 0.9, 9)

    sns.set()
    fig, axs = plt.subplots(len(_training_sets), 1)

    for dataset_id in range(len(_training_sets)):
        dataset_name = 'MONK-{}'.format(dataset_id + 1)

        testset = _testing_sets[dataset_id]
        training_set = _training_sets[dataset_id]

        accuracies = []
        tree_sizes = []
        for fraction in train_val_fractions:
            fraction_accuracies = []
            fraction_tree_sizes = []
            for rep_idx in tqdm(range(num_repetitions),
                                desc='[{}] Testing train-val split: '
                                     '{}'.format(dataset_name, fraction)):

                train_set, val_set = \
                    monkdata.partition(training_set, fraction, seed=rep_idx)
                decision_tree = dt.DecisionTree.train(train_set,
                                                      monkdata.attributes)
                pruned_tree = decision_tree.prune(val_set)
                fraction_accuracies.append(dt.compute_accuracy(pruned_tree,
                                                               testset))
                fraction_tree_sizes.append(pruned_tree.size)

            mean_accuracy = np.mean(fraction_accuracies)
            std_accuracy = np.std(fraction_accuracies)
            min_accuracy = np.min(fraction_accuracies)
            p25_accuracy = np.percentile(fraction_accuracies, 25)
            median_accuracy = np.median(fraction_accuracies)
            p75_accuracy = np.percentile(fraction_accuracies, 75)
            max_accuracy = np.max(fraction_accuracies)

            accuracies.append((
                mean_accuracy,
                std_accuracy,
                min_accuracy,
                p25_accuracy,
                median_accuracy,
                p75_accuracy,
                max_accuracy
            ))
            tree_sizes.append(np.mean(fraction_tree_sizes))

        ax = axs[dataset_id]
        means, stds, mins, p25s, medians, p75s, maxs = \
            map(np.array, zip(*accuracies))

        color='b'
        ax.plot(train_val_fractions, medians, color=color, label='Median')
        ax.fill_between(train_val_fractions,
                        mins,
                        maxs,
                        color=color,
                        label='All',
                        alpha=0.2)
        ax.fill_between(train_val_fractions,
                        p25s,
                        p75s,
                        color=color,
                        label='Interquartile range',
                        alpha=0.5)

        ax.plot(train_val_fractions, means, color=color, label='Mean',
                linestyle='--')
        # ax.fill_between(train_val_fractions,
        #                 means - stds,
        #                 means + stds,
        #                 label='+- standard deviation',
        #                 alpha=0.5)

        ax.set_title(dataset_name)
        ax.set_xlabel('Train-val fraction')
        ax.set_ylabel('Test accuracy')
        ax.legend(loc='lower right')

        # Instantiate the right Y axis for the tree size.
        right_ax = ax.twinx()
        color = 'g'
        right_ax.plot(train_val_fractions, tree_sizes, color=color,
                      label='Tree size')
        right_ax.grid(False)
        right_ax.set_ylabel('Average tree size')

    fig.tight_layout()
    plt.show()
    plt.close()


def main():
    # assignment1()
    # _separator()
    # assignment3()
    # _separator()
    # assignment4()
    # _separator()
    # assignment4p5()
    # _separator()
    # assignment5()
    # _separator()
    # assignment5p5()
    # _separator()
    # assignment7p0()
    # _separator()
    # assignment7p1()
    # _separator()
    assignment7p2()


if __name__ == '__main__':
    main()
