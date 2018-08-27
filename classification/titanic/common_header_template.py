# To support both python2 & 3
from __future__ import division, print_function, unicode_literals

# Common imports
import os
import numpy as np
import pandas as pd

# matplotlib imports
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Path variables
TITANIC_PATH = os.path.join("datasets", "titanic")

# Save fig directory
PROJECT_ROOT_DIR = "."
DIRECTORY = "titanic"

# Set seed value
np.random.seed(42)


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", DIRECTORY, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def load_data(filename, path):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


# Declare train/test data
train_data = load_data("train.csv", TITANIC_PATH)
test_data = load_data("test.csv", TITANIC_PATH)
