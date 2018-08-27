# To support both python 2 & 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# ML imports
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Scientific library imports
from scipy.ndimage.interpolation import shift

# Set seed value
np.random.seed(42)

# Directory variables
PROJECT_ROOT_DIR = "."
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Custom functions


def shift_image(image, dx, dy, new=0):
    # shift image based on x and y ositions
    return shift(image.reshape(28, 28), [dy, dx], cval=new).reshape(784)


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, CURRENT_DIR,
                        "images", fig_id + ".png")
    print("Saving figure", fig_id)


# Fetch test data
mnist = fetch_mldata('MNIST original')

# Split data
X, y = mnist["data"], mnist["target"]
print(X.shape)  # 70000, 784 (28 by 28px)
print(y.shape)  # 70000, 1

# Split for test and training sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Randomize the training sets
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Cross validation on base test estimator
sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train)
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# 5 fold cross validation
knn_clf = KNeighborsClassifier(n_neighbors=3, weights='uniform', n_jobs=-1)

param_grid = {
    # Declare n_neighbors and weights search parameters
    'n_neighbors':  list(range(3, 6)),
    'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(knn_clf, param_grid, verbose=3, cv=5,
                           scoring="accuracy",
                           random_state=42)

grid_search.fit(X_train, y_train)

print(
    "The best estimator hyperparameters: \n" + grid_search.best_estimator_ +
    "\nThe best paramaters: " + grid_search.best_params_ +
    "\nThe best score: " + grid_search.best_score_
)

# predict scores
y_knn_pred = grid_search.predict(X_test)

print("The accuracy: " + accuracy_score(y_test, y_knn_pred))

# Create training set expansions
X_train_expanded = [X_train]
y_train_expanded = [y_train]
print("Shapes before data augmentation: " + X_train.shape + y_train.shape)

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    shifted_images = np.apply_along_axis(shift_image, axis=1, arr=X_train,
                                         dx=dx, dy=dy)
    X_train_expanded.append(shifted_images)
    y_train_expanded.append(y_train)

X_train_expanded = np.concatenate(X_train_expanded)
y_train_expanded = np.concatenate(y_train_expanded)

print("Shapes after data augmentation: " + X_train_expanded + y_train_expanded)
