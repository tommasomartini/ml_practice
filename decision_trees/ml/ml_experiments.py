# -*- coding: utf-8 -*-
'''
Created on Oct 21, 2017

@author: tom
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import random

import matplotlib.pyplot as plt
import numpy as np

from decision_trees.dataset.monkdata import attributes as monk_attributes
from decision_trees.dataset.monkdata import monk1
from decision_trees.dataset.monkdata import monk1test
from decision_trees.dataset.monkdata import monk2
from decision_trees.dataset.monkdata import monk2test
from decision_trees.dataset.monkdata import monk3
from decision_trees.dataset.monkdata import monk3test
from decision_trees.provided.dtree import averageGain, check, allPruned,\
    mostCommon
from decision_trees.provided.dtree import entropy
from decision_trees.provided.dtree import info_gain_ratio
from decision_trees.provided.dtree import select
from decision_trees.provided.dtree import buildTree
from decision_trees.provided.drawtree_qt5 import drawTree


random.seed(34)


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def main():
    datasets = {
        'monk1': monk1,
        'monk2': monk2,
        'monk3': monk3,
    }

    # Split monk1 according to attribute 5.
    print('{} samples ({} true, {} false)'.format(len(monk1),
                                                  len([x for x in monk1 if x.positive]),
                                                  len([x for x in monk1 if not x.positive])))
    
    scnd_lvl_split = {
        1: None,    # node A5=1 -> don't split
        2: 4,   # node A5=2 -> split according to A4
        3: 6,   # node A5=3 -> split according to A6
        4: 1,   # node A5=4 -> split according to A1
    }
    
    for attr_val in monk_attributes[4].values:
        subset_monk1 = select(dataset=datasets['monk1'],
                              attribute=monk_attributes[4],
                              value=attr_val)
        num_true = len([x for x in subset_monk1 if x.positive])
        num_false = len([x for x in subset_monk1 if not x.positive])
        print('\nAttribute 5 value {} ({} samples: {} true, {} false)'.format(attr_val,
                                                           len(subset_monk1),
                                                           num_true,
                                                           num_false))
        next_att = scnd_lvl_split[attr_val]
        if next_att is None:
            print('  Do not split')
            continue
        for attr_val2 in monk_attributes[next_att - 1].values:
            subsubset = select(dataset=subset_monk1,
                              attribute=monk_attributes[next_att - 1],
                              value=attr_val2)
            num_true = len([x for x in subsubset if x.positive])
            num_false = len([x for x in subsubset if not x.positive])
            print('  Attribute {} value {} ({} samples: {} true, {} false)'.format(next_att,
                                                                                   attr_val2,
                                                               len(subsubset),
                                                               num_true,
                                                               num_false))
        
        
#         for i in range(len(monk_attributes)):
#             if i == 4:
#                 continue
#             avg_gain = averageGain(subset_monk1,
#                                    attribute=monk_attributes[i])
#             print('  gain attribute A{}: {}'.format(i + 1, avg_gain))


def main2():
    my_tree = buildTree(dataset=monk1,
                        attributes=monk_attributes,
                        maxdepth=3)
    print(my_tree.size_subtree)
    drawTree(my_tree)


def main3():
    datasets = {
        'monk1': (monk1, monk1test),
        'monk2': (monk2, monk2test),
        'monk3': (monk3, monk3test),
    }
    for set_name, set_split in sorted(datasets.items()):
#         if set_name in ('monk1', 'monk3'):
#             continue
        print(set_name)
        trainset, testset = set_split
        my_tree = buildTree(dataset=trainset,
                            attributes=monk_attributes,
                            maxdepth=2)
#         drawTree(my_tree)
        print(check(my_tree, trainset))
        print(check(my_tree, testset))


def pruning():
    train_val_ratios = np.arange(0.05, 1.0, 0.05) # [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    num_trials = 100
    monk = monk3
    testset = monk3test

    test_acc_before_pruning = {}
    test_acc_after_pruning = {}
    
    train_acc_before = {}
    train_acc_aft = {}
    
    val_acc_before = {}
    val_acc_aft = {}
    
    size_before = {}
    size_after = {}
    
    for train_val_ratio in train_val_ratios:
        test_acc_before = test_acc_before_pruning.setdefault(train_val_ratio, [])
        test_acc_after = test_acc_after_pruning.setdefault(train_val_ratio, [])
        
        t_train_acc_before = train_acc_before.setdefault(train_val_ratio, [])
        t_train_acc_aft = train_acc_aft.setdefault(train_val_ratio, [])
        
        t_val_acc_before = val_acc_before.setdefault(train_val_ratio, [])
        t_val_acc_aft = val_acc_aft.setdefault(train_val_ratio, [])
        
        ssize_bef = size_before.setdefault(train_val_ratio, [])
        ssize_aft = size_after.setdefault(train_val_ratio, [])
        
        for _trial_idx in range(num_trials):
            # Split train and val
            trainset, valset = partition(monk, train_val_ratio)
            my_tree = buildTree(dataset=trainset,
                                attributes=monk_attributes)
            test_acc_before.append(check(my_tree, testset))
            t_train_acc_before.append(check(my_tree, trainset))
            t_val_acc_before.append(check(my_tree, valset))
            
            ssize_bef.append(my_tree.size_subtree)
            # Current best score.
            curr_score = check(my_tree, valset)
            while True:
                # Get all the pruned trees.
                pruned_trees = allPruned(my_tree)
                if len(pruned_trees) == 0:
                    break
                score_map = {}
                for tree_idx, pruned_tree in enumerate(pruned_trees):
                    tree_score = check(pruned_tree, valset)
                    tree_list = score_map.setdefault(tree_score, [])
                    tree_list.append(tree_idx)
                sorted_scores, sorted_tree_idx = zip(*sorted(score_map.items(),
                                                             reverse=True))
                if sorted_scores[0] < curr_score:
                    # None of the pruned trees has better performance.
                    break
                curr_score = sorted_scores[0]
                my_tree = pruned_trees[sorted_tree_idx[0][0]]
            test_acc_after.append(check(my_tree, testset))
            t_train_acc_aft.append(check(my_tree, trainset))
            t_val_acc_aft.append(check(my_tree, valset))
            ssize_aft.append(my_tree.size_subtree)
    
    test_acc_after = [np.mean(v) for k, v in sorted(test_acc_after_pruning.items())]
    test_acc_before = [np.mean(v) for k, v in sorted(test_acc_before_pruning.items())]
    
    avg_train_acc_before = [np.mean(v) for k, v in sorted(train_acc_before.items())]
    avg_train_acc_aft = [np.mean(v) for k, v in sorted(train_acc_aft.items())]
    
    avg_val_acc_before = [np.mean(v) for k, v in sorted(val_acc_before.items())]
    avg_val_acc_aft = [np.mean(v) for k, v in sorted(val_acc_aft.items())]
    
    avg_size_bef = [np.mean(v) for k, v in sorted(size_before.items())]
    avg_size_aft = [np.mean(v) for k, v in sorted(size_after.items())]
    
    plt.figure()
    plt.plot(train_val_ratios, test_acc_after, 'r')
    plt.plot(train_val_ratios, test_acc_before, 'b')
    plt.title('testset')
    plt.legend(('pruned', 'not pruned'))
    
    plt.figure()
    plt.plot(train_val_ratios, avg_train_acc_aft, 'r')
    plt.plot(train_val_ratios, avg_train_acc_before, 'b')
    plt.title('trainset')
    plt.legend(('pruned', 'not pruned'))
    
    plt.figure()
    plt.plot(train_val_ratios, avg_val_acc_aft, 'r')
    plt.plot(train_val_ratios, avg_val_acc_before, 'b')
    plt.title('validation')
    plt.legend(('pruned', 'not pruned'))
    
    plt.figure()
    plt.plot(train_val_ratios, avg_size_aft, 'r')
    plt.plot(train_val_ratios, avg_size_bef, 'b')
    
    plt.show()
    plt.close()


if __name__ == '__main__':
    logging.basicConfig()
    pruning()

