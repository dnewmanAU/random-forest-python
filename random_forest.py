import pandas as pd
from random import seed, randrange
from math import sqrt, log2


def load_data(csv):
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


def cross_validation(data, n_folds):
    """Split dataset into a subset of folds"""
    cv_data = list()
    data_copy = list(data)
    fold_size = int(len(data) / n_folds)
    for _ in range(n_folds):
        fold = list()
        for _ in range(fold_size):
            # get a random data point
            i = randrange(len(data_copy))
            # add it to the fold then remove it from the pool
            fold.append(data_copy.pop(i))
        cv_data.append(fold)
    return cv_data


def random_forest(
    training_set,
    testing_test,
    max_depth,
    min_size,
    sample_size,
    n_trees,
    sqrt_features,
    response_dict,
):
    trees = list()
    for _ in range(n_trees):
        sample = sample_with_replacement(training_set, sample_size)
        build_tree(sample, max_depth, min_size, sqrt_features, response_dict)


def sample_with_replacement(training_set, sample_size):
    """Randomly sample the training set with replacement"""
    sample = list()
    n_samples = round(len(training_set) * sample_size)
    for _ in range(n_samples):
        i = randrange(len(training_set))
        sample.append(training_set[i])
    return sample


def build_tree(sample, max_depth, min_size, sqrt_features, response_dict):
    best_gini = 1.0
    response = list(response_dict.values())
    features = list()
    for _ in range(sqrt_features):
        index = randrange(len(sample[0]) - 1)
        if index not in features:
            features.append(index)
            # [1, 46, 17, 15, 54, 33, 11]
    for index in features:
        for subsample in sample:
            # (index, subsample[index], sample)
            # split the sample in two
            left, right = list(), list()
            for row in sample:
                if row[index] < subsample[index]:
                    left.append(row)
                else:
                    right.append(row)
            split_sample = (left, right)
            gini = gini_impurity(split_sample, response)
            if gini < best_gini:
                best_index = index
                best_subsample = subsample[index]
                best_gini = gini
                best_split = split_sample
    root = {"index": best_index, "subsample": best_subsample, "split": best_split}


def gini_impurity(split_sample, response):
    """Calculate the gini impurity for a randomly split sample"""
    gini = 0.0
    for split in split_sample:
        size = float(len(split))
        # avoid dividing by zero
        if size == 0:
            continue
        score = 0.0
        for val in response:
            p = [row[-1] for row in split].count(val) / size
            score += p * p
        gini += (1.0 - score) * (size / float(len(sum(split_sample, []))))
    return gini


def main():
    seed(100)
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
        random_forest(
            training_set,
            testing_set,
            max_depth,
            min_size,
            sample_size,
            n_trees,
            sqrt_features,
            response_dict,
        )
        break


if __name__ == "__main__":
    main()
