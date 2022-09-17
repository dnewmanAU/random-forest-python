import pandas as pd
from random import seed, randrange
from math import sqrt, log2


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


def get_node(sampled_data, sqrt_features, response_dict):
    """Find the best split (root) for the sampled data"""
    features = list()
    # get a random list of indices from the data
    while len(features) < sqrt_features:
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


def build_tree(node, max_depth, min_size, sqrt_features, response_dict, depth):
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
        node["left"] = get_node(left, sqrt_features, response_dict)
        build_tree(
            node["left"], max_depth, min_size, sqrt_features, response_dict, depth + 1
        )
    # right child
    if len(right) <= min_size:
        node["right"] = terminal_node(right)
    else:
        node["right"] = get_node(right, sqrt_features, response_dict)
        build_tree(
            node["right"], max_depth, min_size, sqrt_features, response_dict, depth + 1
        )


def main():
    seed(5438)
    response_dict, data = load_data("sonar.csv")
    # either square root or log base 2
    sqrt_features = int(sqrt(len(data[0]) - 1))
    log2_features = int(log2(len(data[0]) - 1))

    # Hyperparameters
    n_folds = 5
    max_depth = 10
    min_size = 1
    sample_size = 1.0
    n_trees = 1

    folds = cross_validation(data, n_folds)
    for fold in folds:
        # assign the entire dataset as the training data
        training_set = list(folds)
        # remove the testing set from the training set
        training_set.remove(fold)
        # flatten the training set
        training_set = sum(training_set, [])

        testing_set = list()
        for row in fold:
            row_copy = list(row)
            # remove the response variable
            row_copy.pop()
            testing_set.append(row_copy)
        trees = list()
        for _ in range(n_trees):
            sampled_data = sample_with_replacement(training_set, sample_size)
            root = get_node(sampled_data, sqrt_features, response_dict)
            build_tree(root, max_depth, min_size, sqrt_features, response_dict, depth=1)
        break


if __name__ == "__main__":
    main()
