from random import seed, randrange
import pandas as pd


def load_data(csv):
    # load the csv file into a dataframe
    df = pd.read_csv(csv, header=None)

    # get the response column
    response = df.iloc[:, -1].tolist()

    # extract only the unique response variables
    unique_resp = set(response)

    # encode response variables
    encoded_resp = dict()
    for encode, decode in enumerate(unique_resp):
        print(decode)
        encoded_resp[decode] = encode

    # convert data frame rows into lists
    data = list()
    for row in range(len(df)):
        df.iloc[row, -1] = encoded_resp[df.iloc[row, -1]]
        data.append(df.iloc[row].tolist())

    return [encoded_resp, data]


def main():
    num_features = 0  # either sqrt or log base 2

    encoded_resp, data = load_data("sonar.csv")


if __name__ == "__main__":
    main()
