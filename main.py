import data_set
import id3
import numpy as np

data, labels = data_set.load_data("data\\agaricus-lepiota.data")

# tree display prompt
display_trees = input("Display trees? (y/n): ")
id3_type = input("Algorytm zwykły czy z ruletką? (regular/roulette): ")

# data_set.display_data(data, 10, [i for i in range(23)])
f = 0.1
f = float(input("Podaj rozmiar zbioru treningowego (float): "))

training_data_length = int(len(data) * f)    # 0.5% of data is training data


k = 1;     #numer of repeats
k = int(input("Podaj ilość testów (int): "))

while(k>0):

    #randomize data order
    np.random.shuffle(data)

    training_data = data[:training_data_length, 1:]
    training_labels = data[:training_data_length, 0]

    id3_tree = id3.build_tree(training_labels, training_data, variation=id3_type) #roulette or regular

    if (display_trees == "y"):
        tree_labels = data_set.mushrooms_labels()
        id3_tree.print(tree_labels)

    validation_data = data[training_data_length:, :]
    if len(validation_data) == 0:
        validation_data = data[:, :]

    correct_cases = 0
    all_cases = 0
    not_classified = 0
    for validation_case in validation_data:
        validation = id3_tree.classify(validation_case[1:])
        if validation == validation_case[0]:
            correct_cases += 1
        elif validation == "Cannot classify":
            not_classified += 1
        all_cases += 1

    print("Accuracy over {} cases: {:.2f}%, not classified: {}".format(all_cases, correct_cases / all_cases * 100, not_classified))

    k = k -1