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

    def print(self, labels=None, node=None, tabs=0):
        """
            Prints whole tree in horizontal form
            If labels are specified they are used to replace node/leaf/branch name only in visualization
            Inner names stays the same for implementation reasons
            Labels should be formatted as dictionary of dictionaries, where first dict refers to node/leaf name
            and second dict contains branches names, special key "label" is used to name node/leaf in second
            dictionary, example:
            labels = {'p': {"label": "poisonous"...}, 4: {"label": odor, 'a': "almond", 'n': "none"...}...}

        :param labels: dictionary of dictionaries of names corresponding to nodes and barnches
        :param node: Don't specify
        :param tabs: Don't specify
        :return: None
        """
        if labels is None:
            labels = {}

        if node is None:
            node = self.root

        label = node.label
        # Change label name if specified in dict
        if node.label in labels:
            label = labels[node.label]["label"]

        print('|   ' * tabs, '+---', label, " (", sep='', end='')

        # Print branches
        if node.label in labels:
            branches = []
            for branch in node.branches.keys():
                if branch in labels[node.label]:
                    branches.append(labels[node.label][branch])
                else:
                    branches.append(branch)
            print(*branches, sep=',', end='')
        else:
            print(*node.branches.keys(), sep=',', end='')

        print(')')

        for branch in node.branches.values():
            self.print(labels, branch, tabs + 1)

    def classify(self, case, node=None):
        if node is None:
            node = self.root

        if len(node.branches) == 0:
            return node.label
        else:
            if case[node.label] in node.branches:
                return self.classify(case, node.branches[case[node.label]])
            else:
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
    root = TreeNode(best_attribute)

    for attribute_value in best_attribute_values:
        data_subset_indices = np.argwhere(data[:, best_attribute] == attribute_value)
        data_subset_indices = np.reshape(data_subset_indices, len(data_subset_indices))

        data_subset = data[data_subset_indices]
        labels_subset = labels[data_subset_indices]
        root.branches[attribute_value] = id3(labels_subset, attributes, data_subset)

    return root


def id3_roulette_wheel(labels, attributes, data):
    """
        Implements ID3 Tree generation algorithm and returns fully-built tree.
        This variation of algorithm is a modification of classic id3. Attributes are chosen
        randomly, where probability of choosing an attribute is proportional to it's information
        gain.

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

    # Calculate information gain according to roulette wheel strategy
    cumultative_distribution = np.zeros(len(attributes), dtype=float)
    attribute_values = {}

    for i in range(len(attributes)):
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

        for class_value, attribute_value in zip(labels, data[:, attributes[i]]):
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

        # Save attribute values
        attribute_values[i] = values_table.keys()

        # Calculate cumultative distribution
        if i > 0:
            cumultative_distribution[i] = cumultative_distribution[i - 1] + inf_gain
        else:
            cumultative_distribution[i] = inf_gain

    # Chose attribute according to roulette wheel strategy
    chosen_attribute = 0    # Initial attribute
    temp = np.random.uniform(0, cumultative_distribution[-1])
    for i in range(len(cumultative_distribution)):
        if temp < cumultative_distribution[i]:
            chosen_attribute = i
            break

    attributes.remove(chosen_attribute)
    root = TreeNode(chosen_attribute)

    for attribute_value in attribute_values[chosen_attribute]:
        data_subset_indices = np.argwhere(data[:, chosen_attribute] == attribute_value)
        data_subset_indices = np.reshape(data_subset_indices, len(data_subset_indices))

        data_subset = data[data_subset_indices]
        labels_subset = labels[data_subset_indices]
        root.branches[attribute_value] = id3(labels_subset, attributes, data_subset)

    return root


def build_tree(training_labels, training_data, attributes=None, variation="regular"):
    """
        Builds an ID3 Tree from training data.

    :param training_labels: list of final classification for training data
    :param training_data: data set formatted as 2 dimensional array [[case1], [case2], [case3]...]
    :param attributes: attributes to classify by, if None all attributes in training data are used
    :param variation: variation of algorithm ("regular" for standard id3, "roulette" for roulette wheel based)
    :return: ID3Tree instance
    """

    root = None
    if attributes is None:
        attributes = [i for i in range(len(training_data[0]))]

    if variation == "regular":
        root = id3(training_labels, attributes, training_data)
    elif variation == "roulette":
        root = id3_roulette_wheel(training_labels, attributes, training_data)

    tree = ID3Tree(root)
    return tree
