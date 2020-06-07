from collections import defaultdict
import math

import numpy as np


class TreeNode:
    def __init__(self, label=None):
        self.label = label
        self.branches = dict()


class ID3Tree:
    def __init__(self, root=None):
        self.root = root

    def print_implementation(self, node, tabs=0):
        print('|  ' * tabs, '+--', node.label, " (", sep='', end='')
        print(*node.branches.keys(), sep=',', end='')
        print(')')

        for branch in node.branches.values():
            self.print_implementation(branch, tabs + 1)

    def print(self):
        self.print_implementation(self.root)

    def classify(self, case, node=None):
        if node is None:
            node = self.root

        if len(node.branches) == 0:
            return node.label
        else:
            if case[node.label - 1] in node.branches:
                return self.classify(case, node.branches[case[node.label - 1]])
            else:
                # return self.classify(case, np.random.choice([*node.branches.values()]))
                return "Cannot classify"


def id3(labels, attributes, data):
    """
        Implements ID3 Tree generation algorithm and returns fully-built tree.

    :param labels: list of final labels for training data
    :param attributes: list of available attributes
    :param data: training data set
    :return: TreeNode containing fully built id3 tree
    """
    data_size = len(data)
    if data_size == 0:
        raise RuntimeError("Attempt to build Tree without training data!")

    classes = defaultdict(int)
    for class_value in labels:
        classes[class_value] += 1

    entropy = 0.0

    best_class = list(classes)[0]  # Get first key as starting class
    for curr_class in classes.keys():
        if classes[curr_class] == data_size:
            return TreeNode(curr_class)
        if classes[curr_class] > classes[best_class]:
            best_class = curr_class

        # Calculate data set entropy
        p = classes[curr_class] / data_size
        entropy -= p * math.log2(p)

    if len(attributes) == 0:
        return TreeNode(best_class)

    # Calculate information gain
    best_attribute = None
    best_inf_gain = -math.inf
    best_attribute_values = None

    for attribute in attributes:
        #
        # Values table counts all class_values for all attribute_values
        # Entropy for each attribute can be then calculated by iterating over
        # each class value for given attribute value
        #
        # class/attrib | attrib_val0 | attrib_val1 | attrib_val2 | ...
        #  class_val0  |    val00    |    val10    |    val20    | ...
        #  class_val1  |    val01    |    val11    |    val21    | ...
        #  class_val2  |    val02    |    val12    |    val22    | ...
        #    ...
        values_table = defaultdict(lambda: defaultdict(int))
        values_total = defaultdict(int)

        for class_value, attribute_value in zip(labels, data[:, attribute]):
            values_table[attribute_value][class_value] += 1  # Count class values for given attrib value
            values_total[attribute_value] += 1  # Count total number of attribute value

        # Calculate current information gain
        inf_gain = 0.0
        for attribute_value in values_table.keys():
            # Calculate attribute value entropy
            value_entropy = 0.0
            for class_value in values_table[attribute_value].values():
                p = class_value / values_total[attribute_value]
                value_entropy -= p * math.log2(p)

            inf_gain += (values_total[attribute_value] / data_size) * value_entropy

        inf_gain = entropy - inf_gain

        if inf_gain > best_inf_gain:
            best_attribute = attribute
            best_attribute_values = values_table.keys()
            best_inf_gain = inf_gain

    attributes.remove(best_attribute)
    root = TreeNode(best_attribute + 1)

    for attribute_value in best_attribute_values:
        data_subset_indices = np.argwhere(data[:, best_attribute] == attribute_value)
        data_subset_indices = np.reshape(data_subset_indices, len(data_subset_indices))

        data_subset = data[data_subset_indices]
        labels_subset = labels[data_subset_indices]
        root.branches[attribute_value] = id3(labels_subset, attributes, data_subset)

    return root


def build_tree(training_data, attribute_labels=None):
    """
        Builds an ID3 Tree from training data

    :param training_data:       data set formatted as 2 dimensional array [[case1], [case2], [case3]...]
    :param attribute_labels:    dictionary of labels corresponding to attribute index (eg. {1: "Outlook"})
    :return: ID3Tree instance
    """

