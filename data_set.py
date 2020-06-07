from collections import defaultdict
import numpy as np


def load_data(path, labels=False):
    """
        Loads data from .csv file

    :param path: path to formatted data text file
    :param labels: specifies if file defines labels in first line
    :return: loaded data
    """

    file = open(path, "r")
    data_set = []

    labels_list = None
    if labels:
        labels_list = file.readline().split(',')

    for line in file:
        attributes = line.split(',')
        attributes[-1] = attributes[-1].replace('\n', '')  # Remove new line character from last attribute
        data_set.append(np.array([*attributes]))

    return np.array(data_set), labels_list


def display_data(data, lines: int = 10, labels=None):
    """
        Display loaded data set

    :param data:    data loaded by load_data
    :param lines:   number of lines to show
    :param labels:  data columns labels
    :return:        None
    """

    if labels is not None:
        print(*labels, sep=',')

    if lines > len(data):
        lines = len(data)

    for i in range(lines):
        print(*data[i], sep=',')
