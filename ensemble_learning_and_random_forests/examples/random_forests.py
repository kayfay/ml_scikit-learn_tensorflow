# Support for Python 2 and 3
from __future__ import print_function, unicode_literals, division

# Common Imports
import os
import numpy as np

# ML imports
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Graph Imports
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Directory Config
PROJECT_ROOT_DIR = '.'

# Define Functions


def image_path(fig_id):
    if not os.path.exists('images'):
        os.makedirs('images')
    return os.path.join(PROJECT_ROOT_DIR, 'images', fig_id)


def save_fig(fig_id, tight_layout=True):
    print("Saving ", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)


# Load data
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(iris["data"], iris["target"])

# Feature Importances
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)

