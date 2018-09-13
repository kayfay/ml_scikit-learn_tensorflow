# For python 2 and 3 compatibly
from __future__ import print_function, unicode_literals, division

# Common Imports
import numpy as np
from scipy.stats import mode

# ML Imports
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone

# Graph Imports
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Generate a moons dataset split, train, test
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Select GridSearch Parameters
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}

# Train with GridSearch CV
grid_search_cv = GridSearchCV(
    DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1)
grid_search_cv.fit(X_train, y_train)

print("The best tuned model\n", grid_search_cv.best_estimator_)

y_pred = grid_search_cv.predict(X_test)
print("Tree accuracy score: ", accuracy_score(y_test, y_pred))

# Grow a forest
n_trees = 1000
n_instances = 100

mini_sets = []

# Generate 1000 random subsets of 100 random instances
rs = ShuffleSplit(
    n_splits=n_trees, test_size=len(X_train) - n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

# Train trees on subsets
forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)

    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Gather frequent modes from trees for a majorty vote over the sets
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

Y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

print("Forest accuracy score: ",
      accuracy_score(y_test, Y_pred_majority_votes.reshape([-1])))
