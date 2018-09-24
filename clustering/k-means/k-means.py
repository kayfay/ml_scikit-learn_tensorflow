# Python 2 and 3 support
from __future__ import division, unicode_literals, print_function

# Common Imports
import os
import numpy as np
import time

# ML Imports
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

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


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker='o',
        s=30,
        linewidths=8,
        color=circle_color,
        zorder=10,
        alpha=0.9)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker='x',
        s=50,
        linewidths=50,
        color=cross_color,
        zorder=11,
        alpha=1)


def plot_decision_boundaries(clusterer,
                             X,
                             resolution=1000,
                             show_centroids=True,
                             show_xlabels=True,
                             show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], resolution),
        np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(
        Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel2")
    plt.contour(
        Z,
        extent=(mins[0], maxs[0], mins[1], maxs[1]),
        linewidths=1,
        colors="k")
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)

    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')


def plot_cluster_comparison(clusterer1,
                            clusterer2,
                            X,
                            title1=None,
                            title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)


blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8],
                         [-2.8, 1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(
    n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=2)

# Plot the data
plt.figure(figsize=(8, 4))
plot_clusters(X)
save_fig("blobs_diagram")
plt.show()

# Fit and predict
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

# Each instance applied to one of 5 clusters
y_pred  # kmeans.labels_ array([4, 0, 1 ... 2, 1, 0], dtype=int32)

# Compare true and false
y_pred is kmeans.labels_  # True

# The following 5 centroids (cluster centers) estimates
kmeans.cluster_centers_  # 5 x 2

# Predict from new instances
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)

# Compare Euclidian distance between centroids
soft_clustering = kmeans.transform(X_new)
distances = np.tile(X_new, (1, k)).reshape(-1, k, 2) - kmeans.cluster_centers_
np.allclose(soft_clustering, np.linalg.norm(distances, axis=2))  # True

# K-Means algorithm
kmeans_iter1 = KMeans(
    n_clusters=5,
    init="random",
    n_init=1,
    algorithm="full",
    max_iter=1,
    random_state=2)
kmeans_iter2 = KMeans(
    n_clusters=5,
    init="random",
    n_init=1,
    algorithm="full",
    max_iter=2,
    random_state=2)
kmeans_iter3 = KMeans(
    n_clusters=5,
    init="random",
    n_init=1,
    algorithm="full",
    max_iter=3,
    random_state=2)

kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)

# Plot a voronoi_diagram
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)

save_fig("voronoi_diagram")
plt.show()

# Plot K-Means algorithm
plt.figure(figsize=(10, 8))
plt.subplot(321)
plot_data(X)
plot_centroids(
    kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom='off')
plt.title("Update the centroids (initially randomly)", fontsize=14)

plt.subplot(322)
plot_decision_boundaries(
    kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
plt.title("Label the instances", fontsize=14)

plt.subplot(323)
plot_decision_boundaries(
    kmeans_iter1, X, show_centroids=False, show_xlabels=False)
plot_centroids(kmeans_iter2.cluster_centers_)

plt.subplot(324)
plot_decision_boundaries(
    kmeans_iter2, X, show_xlabels=False, show_ylabels=False)

plt.subplot(325)
plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
plot_centroids(kmeans_iter3.cluster_centers_)

plt.subplot(326)
plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)

save_fig("kmeans_algorithm_diagram")
plt.show()

# K-Means variability
kmeans_rnd_init1 = KMeans(
    n_clusters=5, init="random", n_init=1, algorithm="full", random_state=11)
kmeans_rnd_init2 = KMeans(
    n_clusters=5, init="random", n_init=1, algorithm="full", random_state=19)

plot_cluster_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X, "Solution 1",
                        "Solution 2 (w/ diff random init)")

save_fig("kmeans_vaiability_diagram")
plt.show()

# K-Means Inertia
# Metrics for an unsupervised task
# inertia is the distance between centroids

# Compare distances and inertia
X_distances = kmeans.transform(X)
sums_squared = np.sum(X_distances[np.arange(len(X_distances)), kmeans.labels_]
                      **2)
np.allclose(sums_squared, kmeans.inertia_)  # True

# Predictors scoring method for negative inertia
kmeans.score(X)  # -211.59853

# Minimizing inertia with random initalizations
# Current inertias comparing n_init=1
inertias = [
    kmeans.inertia_, kmeans_rnd_init1.inertia_, kmeans_rnd_init2.inertia_
]  # 211, 223, 237

kmeans_rnd_10_inits = KMeans(
    n_clusters=5, init='random', n_init=10, algorithm="full", random_state=11)
kmeans_rnd_10_inits.fit(X)

plt.figure(figsize=(8, 4))
plt.title("Best of 10")
plot_decision_boundaries(kmeans_rnd_10_inits, X)
plt.show()

# Show speed differenecs between algorithm types
# Eklan's K-Means for non-sparse speed up
t0 = time.time()
[KMeans(algorithm='elkan').fit(X)] * 50
t1 = time.time()

# Full for dense data
t2 = time.time()
[KMeans(algorithm="full").fit(X)] * 50
t3 = time.time()
print("Elkan : {:.2f}s".format(t1 - t0))
print("Full : {:.2f}s".format(t3 - t2))
