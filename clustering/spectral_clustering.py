# Python 2 and 3 support
from __future__ import division, unicode_literals, print_function

# Common Imports
import os
import numpy as np

# ML Imports
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

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


def plot_spectral_clustering(sc,
                             X,
                             size,
                             alpha,
                             show_xlabels=True,
                             show_ylabels=True):
    plt.scatter(
        X[:, 0],
        X[:, 1],
        marker='o',
        s=size,
        c='gray',
        cmap="Paired",
        alpha=alpha)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=30, c='w')
    plt.scatter(
        X[:, 0], X[:, 1], marker=".", s=10, c=sc.labels_, cmap="Paired")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
    plt.title("RBF gamma={}".format(sc.gamma), fontsize=14)


if __name__ == "__main__":
    X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

    sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)
    sc1.fit(X)

    sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)
    sc2.fit(X)

    print(np.percentile(sc1.affinity_matrix_, 95))

    plt.figure(figsize=(9, 3.2))

    plt.subplot(121)
    plot_spectral_clustering(sc1, X, size=500, alpha=0.1)

    plt.subplot(122)
    plot_spectral_clustering(sc2, X, size=4000, alpha=0.01, show_ylabels=False)

    plt.show()
