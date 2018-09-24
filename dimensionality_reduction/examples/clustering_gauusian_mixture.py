# When clusters are closly grouped together Gaussian Mixture
# transform seperation
# Python 2 and 3 support
from __future__ import division, unicode_literals, print_function

# Common Imports
import os
import numpy as np

# ML Imports
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

# Graph Imports
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Config
PROJECT_ROOT_DIR = '.'


# Declare Functions
def image_path(fig_id):
    if not os.path.exists('images'):
        os.makedirs('images')
    return os.path.join(PROJECT_ROOT_DIR, 'images', fig_id)


def save_fig(fig_id, tight_layout=True):
    print("Saving", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format="png", dpi=300)


# Create datasets
data = load_iris()
X = data.data
y = data.target
data.target_names

plt.figure(figsize=(9, 3.5))
plt.subplot(121)
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(fontsize=12)

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c="k", marker=".")
plt.xlabel("Petal length", fontsize=14)
plt.tick_params(labelleft="off")

save_fig("classification_vs_clustering_diagram")
plt.show()

# Seperate clusters with Gaussian mixture
y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)
mapping = np.array([2, 0, 1])
y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

plt.plot(X[y_pred == 0, 0], X[y_pred == 0, 1], "yo", label="Cluster 1")
plt.plot(X[y_pred == 1, 0], X[y_pred == 1, 1], "bs", label="Cluster 2")
plt.plot(X[y_pred == 2, 0], X[y_pred == 2, 1], "g^", label="Cluster 3")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("petal width", fontsize=14)
plt.legend(loc="upper right", fontsize=12)
plt.show()

total_accuracy = np.sum(y_pred == y)
percent_accuracy = total_accuracy / len(y_pred)

print("Total Correct:", total_accuracy)
print("Correct Total: {:.2%}".format(percent_accuracy) )
