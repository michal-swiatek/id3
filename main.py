import data_set
import id3
import numpy as np

data, labels = data_set.load_data("data\\agaricus-lepiota.data")

# data_set.display_data(data, 10, [i for i in range(23)])

training_data_length = int(len(data) * 0.05)    # 0.5% of data is training data

k = 10;     #numer of repeats

while(k>0):

    #randomize data order
    np.random.shuffle(data)

    training_data = data[:training_data_length, 1:]
    training_labels = data[:training_data_length, 0]

    id3_tree = id3.build_tree(training_labels, training_data)

    tree_labels = data_set.mushrooms_labels()

    id3_tree.print(tree_labels)

    validation_data = data[training_data_length:, :]
    if len(validation_data) == 0:
        validation_data = data[:, :]

    correct_cases = 0
    all_cases = 0
    for validation_case in validation_data:
        if validation_case[0] == id3_tree.classify(validation_case[1:]):
            correct_cases += 1
        all_cases += 1

    print("Accuracy over {} cases: {:.2f}%".format(all_cases, correct_cases / all_cases * 100))

    k = k -1