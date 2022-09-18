import pandas as pd
from random import seed, randrange
from math import sqrt, log2
from time import time


def load_data(csv):
    """Load a csv dataset"""
    # load the csv file into a dataframe
    df = pd.read_csv(csv, header=None)
    # get the response column
    response = df.iloc[:, -1].tolist()
    # extract the unique response variables
    unique_resp = set(response)
    # encode the response variables
    response_dict = dict()
    for encode, decode in enumerate(unique_resp):
        response_dict[decode] = encode
    # convert dataframe rows into nested lists
    data = list()
    for row in range(len(df)):
        df.iloc[row, -1] = response_dict[df.iloc[row, -1]]  # encode response
        data.append(df.iloc[row].tolist())
    return [response_dict, data]


def cross_validation(data, num_folds):
    """Split dataset into a subset of folds"""
    cv_data = list()
    data_copy = list(data)
    fold_size = int(len(data) / num_folds)
    for _ in range(num_folds):
        fold = list()
        for _ in range(fold_size):
            # get a random data point
            i = randrange(len(data_copy))
            # add it to the fold then remove it from the pool
            fold.append(data_copy.pop(i))
        cv_data.append(fold)
    return cv_data


def sample_with_replacement(training_set, sample_size):
    """Randomly sample the training set with replacement"""
    sample = list()
    num_samples = round(len(training_set) * sample_size)
    for _ in range(num_samples):
        i = randrange(len(training_set))
        sample.append(training_set[i])
    return sample


def gini_impurity(split_sample, response):
    """Calculate the gini impurity for a randomly split sample"""
    gini = 0.0
    for split in split_sample:
        size = float(len(split))
        # avoid dividing by zero
        if size == 0:
            continue
        score = 0.0
        # calculate the score of the split for each response variable
        for val in response:
            p = [row[-1] for row in split].count(val) / size
            score += p * p
        gini += (1.0 - score) * (size / float(len(sum(split_sample, []))))
    return gini


def get_node(sampled_data, num_features, response_dict):
    """Find the best split (root) for the sampled data"""
    features = list()
    # get a random list of indices from the data
    while len(features) < num_features:
        index = randrange(len(sampled_data[0]) - 1)
        if index not in features:
            features.append(index)
    response = list(response_dict.values())
    best_gini = 1.0
    # get the gini impurity for each
    for index in features:
        for sample in sampled_data:
            # split the sample in two
            left_sample, right_sample = list(), list()
            for row in sampled_data:
                if row[index] < sample[index]:
                    left_sample.append(row)
                else:
                    right_sample.append(row)
            split_sample = (left_sample, right_sample)
            gini = gini_impurity(split_sample, response)
            # the lower the gini, the better the split
            if gini < best_gini:
                best_index = index
                best_sample = sample[index]
                best_split = split_sample
                best_gini = gini
    root = {"index": best_index, "sample": best_sample, "split": best_split}
    return root


def build_tree(node, max_depth, min_size, num_features, response_dict, depth):
    def terminal_node(node):
        outcomes = list()
        for row in node:
            outcomes.append(row[-1])
        return max(set(outcomes), key=outcomes.count)

    left, right = node["split"]
    del node["split"]
    # no split
    if not left or not right:
        node["left"] = node["right"] = terminal_node(left + right)
        return
    # keep branching until max depth is reached
    if depth >= max_depth:
        node["left"] = terminal_node(left)
        node["right"] = terminal_node(right)
        return
    # left child
    if len(left) <= min_size:
        node["left"] = terminal_node(left)
    else:
        node["left"] = get_node(left, num_features, response_dict)
        build_tree(
            node["left"], max_depth, min_size, num_features, response_dict, depth + 1
        )
    # right child
    if len(right) <= min_size:
        node["right"] = terminal_node(right)
    else:
        node["right"] = get_node(right, num_features, response_dict)
        build_tree(
            node["right"], max_depth, min_size, num_features, response_dict, depth + 1
        )


def bagging(trees, test_row):
    def predict(node, test_row):
        # change sample to value
        if test_row[node["index"]] < node["sample"]:
            if isinstance(node["left"], dict):
                return predict(node["left"], test_row)
            else:
                return node["left"]
        else:
            if isinstance(node["right"], dict):
                return predict(node["right"], test_row)
            else:
                return node["right"]

    predictions = list()
    for tree in trees:
        predictions.append(predict(tree, test_row))
    return max(set(predictions), key=predictions.count)


def main():
    response_dict, data = load_data("sonar.csv")
    # either square root or log base 2
    # sqrt_features = int(sqrt(len(data[0]) - 1))
    log2_features = int(log2(len(data[0]) - 1))

    # Hyperparameters
    num_folds = 8
    max_depth = 7
    min_size = 3
    sample_size = 1.0
    num_trees = 10

    start_time = time()

    folds = cross_validation(data, num_folds)
    accuracy = list()
    for fold in folds:
        # assign the entire dataset as the training data
        training_set = list(folds)
        # remove the testing set from the training set
        training_set.remove(fold)
        # flatten the training set
        training_set = sum(training_set, [])

        testing_set = list()
        actual_response = list()
        for row in fold:
            row_copy = list(row)
            actual_response.append(row_copy[-1])
            # remove the response variable
            row_copy.pop()
            testing_set.append(row_copy)
        trees = list()
        for _ in range(num_trees):
            sampled_data = sample_with_replacement(training_set, sample_size)
            # get the best root for the sampled data
            root = get_node(sampled_data, log2_features, response_dict)
            # build the tree from the root
            build_tree(root, max_depth, min_size, log2_features, response_dict, depth=1)
            trees.append(root)
        predictions = list()
        for test_row in testing_set:
            predictions.append(bagging(trees, test_row))
        correct = 0
        for i in range(len(actual_response)):
            if actual_response[i] == predictions[i]:
                correct += 1
        accuracy.append(correct / float(len(actual_response)) * 100.0)

    print("Trees: %d" % num_trees)
    print("Accuracy: %s" % accuracy)
    print("Mean Accuracy: %.3f%%" % (sum(accuracy) / float(len(accuracy))))

    end_time = time() - start_time
    print("\nRuntime: " + "{:.2f}".format(end_time) + " seconds")


if __name__ == "__main__":
    main()
