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

    # convert dataframe rows into a list of lists
    data = list()
    for row in range(len(df)):
        df.iloc[row, -1] = response_dict[df.iloc[row, -1]]  # encode response
        data.append(df.iloc[row].tolist())

    return [response_dict, data]


def main():
    response_dict, data = load_data("data.csv")
    # either square root or log base 2
    sqrt_features = sqrt(len(data[0]) - 1)
    log2_features = log2(len(data[0]) - 1)


if __name__ == "__main__":
    main()
