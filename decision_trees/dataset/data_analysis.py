# -*- coding: utf-8 -*-
'''
Created on Oct 20, 2017

@author: tom
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from decision_trees.dataset.monkdata import attributes as monk_attributes
from decision_trees.dataset.monkdata import monk1
from decision_trees.dataset.monkdata import monk1test
from decision_trees.dataset.monkdata import monk2
from decision_trees.dataset.monkdata import monk2test
from decision_trees.dataset.monkdata import monk3
from decision_trees.dataset.monkdata import monk3test
from decision_trees.provided.dtree import averageGain
from decision_trees.provided.dtree import entropy
from decision_trees.provided.dtree import info_gain_ratio
from decision_trees.provided.dtree import select


def datasets_entropy(datasets):
    for dataset_name, dataset in sorted(datasets.items()):
        set_entropy = entropy(dataset)
        print('Entropy({}) = {}'.format(dataset_name, set_entropy))


def info_gains(datasets):
    for dataset_name, dataset in sorted(datasets.items()):
        print('Dataset {}'.format(dataset_name))
        for i in range(len(monk_attributes)):
            avg_gain = averageGain(dataset,
                                   attribute=monk_attributes[i])
            print('  gain attribute {}: {}'.format(i, avg_gain))


def info_gain_ratios(datasets):
    for dataset_name, dataset in sorted(datasets.items()):
        print('Dataset {}'.format(dataset_name))
        for i in range(len(monk_attributes)):
            igr = info_gain_ratio(dataset,
                                  attribute=monk_attributes[i])
            print('  gain ratio attribute {}: {}'.format(i, igr))


def main():
    datasets = {
        'monk1': monk1,
        'monk2': monk2,
        'monk3': monk3,
    }

    datasets_entropy(datasets)
    info_gains(datasets)
    info_gain_ratios(datasets)


if __name__ == '__main__':
    logging.basicConfig()
    main()
