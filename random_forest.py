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
    """Split dataset into a subset of folds."""
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


def main():
    seed(100)
    response_dict, data = load_data("data.csv")
    # either square root or log base 2
    sqrt_features = int(sqrt(len(data[0]) - 1))
    log2_features = int(log2(len(data[0]) - 1))

    # Hyperparameters
    n_folds = 5
    max_depth = 10

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


if __name__ == "__main__":
    main()
