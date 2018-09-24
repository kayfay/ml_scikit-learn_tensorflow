"""
Perform principle components analysis on a dataset and compare it to
a non decomposed model
"""

# Python 2 and 3 support
from __future__ import division, unicode_literals, print_function

# Common Imports
import numpy as np
import os
import time

# ML Imports
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Import Graph
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Config
PROJECT_ROOT_DIR = "."


# Declare Functions
def image_path(fig_id):
    if not os.path.exists('images'):
        os.makedirs('images')
    return os.path.join(PROJECT_ROOT_DIR, 'images', fig_id)


def save_fig(fig_id, tight_layout=True):
    print("Saving", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format=r'png', dpi=300)


# Get data
mnist = fetch_mldata('MNIST original')

# Split data
X_train = mnist['data'][:60000]
y_train = mnist['target'][:60000]

X_test = mnist['data'][60000:]
y_test = mnist['target'][60000:]

# Train CLF
rnd_clf = RandomForestClassifier(random_state=42)

t0 = time.time()
rnd_clf.fit(X_train, y_train)
t1 = time.time()

y_pred = rnd_clf.predict(X_test)

# Report metrics
print("Training time: {:.2f}s".format(t1 - t0))
print(rnd_clf.__class__.__name__, "Score", accuracy_score(y_test, y_pred))

# Dimensionality Reduction
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)

# Train on PCA dimensionality reduction
rnd_clf2 = RandomForestClassifier(random_state=42)
t2 = time.time()
rnd_clf2.fit(X_train_reduced, y_train)
t3 = time.time()

X_test_reduced = pca.transform(X_test)
y_pred = rnd_clf2.predict(X_test_reduced)

# Report new metrics
print("Training time: {:.2f}s".format(t3 - t2))
print(rnd_clf2.__class__.__name__, "Score", accuracy_score(y_test, y_pred))

# Compare to softmax regression
log_clf = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", random_state=42)
t4 = time.time()
log_clf.fit(X_train, y_train)
t5 = time.time()
y_pred = log_clf.predict(X_test)

print("Training time: {:.2f}s".format(t5 - t4))
print(log_clf.__class__.__name__, "Score", accuracy_score(y_test, y_pred))

# Compare softmax regression with PCA decomposition
log_clf2 = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", random_state=42)
t6 = time.time()
log_clf2.fit(X_train_reduced, y_train)
t7 = time.time()
y_pred = log_clf2.predict(X_test_reduced)

print("Training time: {:.2f}s".format(t7 - t6))
print(log_clf.__class__.__name__, "Score", accuracy_score(y_test, y_pred))
