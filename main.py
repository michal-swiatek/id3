import data_set
import id3

data, labels = data_set.load_data("data\\agaricus-lepiota.data")

# data_set.display_data(data, 10, [i for i in range(23)])

training_data_length = int(len(data) * 0.75)    # 75% of data is training data
training_data = data[:training_data_length, 1:]
training_labels = data[:training_data_length, 0]

# id3_tree = id3.id3(0, [5], data[:, :])
id3_tree = id3.id3(training_labels, [i for i in range(22)], training_data)
id3_tree = id3.ID3Tree(id3_tree)

id3_tree.print()

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
