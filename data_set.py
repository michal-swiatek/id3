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


def mushrooms_labels():
    labels = {
        # Final classes
        'p': {"label": "poisonous"},
        'e': {"label": "edible"},

        # Attributes
        0: {
            "label": "cap-shape",
            'b': "bell",
            'c': "conical",
            'x': "convex",
            'f': "flat",
            'k': "knobbed",
            's': "sunken"
        },

        1: {
            "label": "cap-surface",
            'f': "fibrous",
            'g': "grooves",
            'y': "scaly",
            's': "smooth"
        },

        2: {
            "label": "cap-color",
            'n': "brown",
            'b': "buff",
            'c': "cinnamon",
            'g': "gray",
            'r': "green",
            'p': "pink",
            'u': "purple",
            'e': "red",
            'w': "white",
            'y': "yellow"
        },

        3: {
            "label": "bruises?",
            't': "bruises",
            'f': "no"
        },

        4: {
            "label": "odor",
            'a': "almond",
            'l': "anise",
            'c': "creosote",
            'y': "fishy",
            'f': "foul",
            'm': "musty",
            'n': "none",
            'p': "pungent",
            's': "spicy"
        },

        5: {
            "label": "gill-attachment",
            'a': "attached",
            'd': "descending",
            'f': "free",
            'n': "notched"
        },

        6: {
            "label": "gill-spacing",
            'c': "close",
            'w': "crowded",
            'd': "distant"
        },

        7: {
            "label": "gill-size",
            'b': "broad",
            'n': "narrow"
        },

        8: {
            "label": "gill-color",
            'k': "black",
            'n': "brown",
            'b': "buff",
            'h': "chocolate",
            'g': "gray",
            'r': "green",
            'o': "orange",
            'p': "pink",
            'u': "purple",
            'e': "red",
            'w': "white",
            'y': "yellow"
        },

        9: {
            "label": "stalk-shape",
            'e': "enlarging",
            't': "tapering"
        },

        10: {
            "label": "stalk-root",
            'b': "bulbous",
            'c': "club",
            'u': "cup",
            'e': "equal",
            'z': "rhizomorphs",
            'r': "rooted",
            '?': "missing"
        },

        11: {
            "label": "stalk-surface-above-ring",
            'f': "fibrous",
            'y': "scaly",
            'k': "silky",
            's': "smooth"
        },

        12: {
            "label": "stalk-surface-below-ring",
            'f': "fibrous",
            'y': "scaly",
            'k': "silky",
            's': "smooth"
        },

        13: {
            "label": "stalk-color-above-ring",
            'n': "brown",
            'b': "buff",
            'c': "cinnamon",
            'g': "gray",
            'o': "orange",
            'p': "pink",
            'e': "red",
            'w': "white",
            'y': "yellow"
        },

        14: {
            "label": "stalk-color-below-ring",
            'n': "brown",
            'b': "buff",
            'c': "cinnamon",
            'g': "gray",
            'o': "orange",
            'p': "pink",
            'e': "red",
            'w': "white",
            'y': "yellow"
        },

        15: {
            "label": "veil-type",
            'p': "partial",
            'u': "universal"
        },

        16: {
            "label": "veil-color",
            'n': "brown",
            'o': "orange",
            'w': "white",
            'y': "yellow"
        },

        17: {
            "label": "ring-number",
            'n': "none",
            'o': "one",
            't': "two"
        },

        18: {
            "label": "ring-type",
            'c': "cobwebby",
            'e': "evanescent",
            'f': "flaring",
            'l': "large",
            'n': "none",
            'p': "pendant",
            's': "sheathing",
            'z': "zone"
        },

        19: {
            "label": "spore-print-color",
            'k': "black",
            'n': "brown",
            'b': "buff",
            'h': "chocolate",
            'r': "green",
            'o': "orange",
            'u': "purple",
            'w': "white",
            'y': "yellow"
        },

        20: {
            "label": "population",
            'a': "abundant",
            'c': "clustered",
            'n': "numerous",
            's': "scattered",
            'v': "several",
            'y': "solitary"
        },

        21: {
            "label": "habitat",
            'g': "grasses",
            'l': "leaves",
            'm': "meadows",
            'p': "paths",
            'u': "urban",
            'w': "waste",
            'd': "woods"
        }
    }

    return labels
