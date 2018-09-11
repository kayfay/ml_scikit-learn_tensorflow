# Using an SVM to classify digits
# Python 2 and 3 support
from __future__ import print_function, unicode_literals, division

# Common Imports
import os
import numpy as np

# ML Imports
from sklearn.datasets import fetch_mldata
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

# Stats imports
from scipy.stats import reciprocal, uniform

# Graph Imports
import matplotlib.pyplot as plt

# Directory Config
PROJECT_ROOT_DIR = '.'


# Declare Functions
def save_fig(fig_id, tight_layout=True):
    if not os.path.exists('images'):
        os.makedirs('images')
    path = os.path.join('PROJECT_ROOT_DIR', 'images', fig_id + '.png')
    print("Saving figure", fig_id)
    if tight_layout():
        plt.tight_layout()
    plt.savefig(path, format=r'png', dpi=300)


# Set data
np.random.seed(42)
mnist = fetch_mldata("MNIST original")
X = mnist["data"]
y = mnist["target"]

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

rnd_idx = np.random.permutation(60000)
X_train = X_train[rnd_idx]
y_train = y_train[rnd_idx]

# Model
lin_clf = LinearSVC(random_state=42)
#  lin_clf.fit(X_train, y_train) # Base model
#  y_pred = lin_clf.predict(X_train)
#  accuracy_score(y_train, y_pred) .83

# Scale data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

#  lin_clf.fit(X_train_scaled, y_train) # Scaled model
#  y_pred = lin_clf.predict(X_train_scaled)
#  accuracy_score(y_train, y_pred) .92

# Model with RBF kernel function
svm_clf = SVC(decision_function_shape="ovr")

#  svm_clf.fit(X_train_scaled[:10000], y_train[:10000])
#  y_pred = svm_clf.predict(X_train_scaled)
#  accuracy_score(y_train, y_pred) .95

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(
    svm_clf, param_distributions, n_iter=10, verbose=2)
rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

# Training prediction
y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
accuracy_score(y_train, y_pred)  # .99

# Test prediction
y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
test_set_score = accuracy_score(y_test, y_pred)  # .97

print("Score on test set:", test_set_score)

