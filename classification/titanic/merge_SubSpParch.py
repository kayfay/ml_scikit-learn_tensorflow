import os
import numpy as np
import pandas as pd


def load_data(filename, path):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


def col_sum(data, new, col1, col2):
    data[str(new)] = data[str(col1)] + data[str(col2)]
    return data.drop([str(col1), str(col2)], axis=1)


TITANIC_PATH = os.path.join("datasets", "titanic")

train_data = load_data("train.csv", TITANIC_PATH)
test_data = load_data("test.csv", TITANIC_PATH)

train_data = col_sum(train_data, 'FamMems', 'SibSp', 'Parch')
test_data = col_sum(test_data, 'FamMems', 'SibSp', 'Parch')
