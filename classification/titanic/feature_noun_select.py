# To support both python 2 & 3
from __future__ import division, print_function, unicode_literals

# Common imports
import os
import numpy as np
import pandas as pd
import nltk

TITANIC_PATH = os.path.join('datasets', 'titanic')

PROJECT_ROOT_DIR = '.'
DIRECTORY = 'titanic'

noun_rows = list()


def load_data(filename, path):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


def count_rows(data, row_name):
    n_row = data.count()[str(row_name)]
    for row_index in n_row:
        nltk.word_tokenize.pos_tag()
        noun_rows.append(row_index)


def row_token_dict(data):
    # take data as dict
    names = dict(data.loc[:, 'Name'])
    # tokenize values
    names.values

    # build dict: index & row_tokens


X_train = load_data('train.csv', TITANIC_PATH)

# Count the row instances in the Name feature
total_rows = X_train.count()['Name']
print(X_train.count)

for row_index in total_rows:
    print(row_index)

# Tag words for nouns from a dataframe row
print('Name: ', row)
tokens = nltk.word_tokenize(row)
print('Tokens: ', tokens)
tagged = nltk.pos_tag(tokens)
print('Tagged: ', tagged)


for word in tagged:
    if word[1][0] == 'N':
        noun_rows.append(row_index)

print("Current Indexes: ", nouns_rows)
