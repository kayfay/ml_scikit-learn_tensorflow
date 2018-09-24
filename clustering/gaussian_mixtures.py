# Python 2 and 3 support
from __future__ import division, unicode_literals, print_function

# Common Imports
import os
import numpy as np

# ML Imports
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Graph Imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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


def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], resolution),
        np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(
        xx,
        yy,
        Z,
        norm=LogNorm(vmin=1.0, vmax=30.0),
        levels=np.logspace(0, 2, 12))
    plt.contour(
        xx,
        yy,
        Z,
        norm=LogNorm(vmin=1.0, vmax=30.0),
        levels=np.logspace(0, 2, 12),
        linewidths=1,
        colors='k')
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, linewidths=2, colors='r', linestyles='dashed')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)


def compare_gaussian_mixture(gm1, gm2, X):
    plt.figure(figsize=(9, 4))

    plt.subplot(121)
    plot_gaussian_mixture(gm1, X)
    #plt.title(
    #    'covariance_type="{}"'.format(gm1.get_params()['covariance_type']),
    #    fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(gm2, X, show_ylabels=False)
    #plt.title(
    #    'covariance_type="{}"'.format(gm2.get_params()['convariance_type']),
    #    fontsize=14)


# Dataset
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

# Train model
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)
print("EM Estimates", gm.weights_)
print("EM Means", gm.means_)
print("EM Covariances", gm.covariances_)
print("Convergance, and iterations", gm.converged_, gm.n_iter_)
print("Hard clustering predictions", gm.predict(X))
print("Hard clustering probabilities", gm.predict_proba(X))

# Generative examples
X_new, y_new = gm.sample(6)

# Log probability density function (PDF)
print("PDF for X values", gm.score_samples(X))

# Check PDF integration to 1
resolution = 100
grid = np.arange(-10, 10, 1 / resolution)
xx, yy = np.meshgrid(grid, grid)
X_full = np.vstack([xx.ravel(), yy.ravel()]).T

pdf = np.exp(gm.score_samples(X_full))
pdf_probas = pdf * (1 / resolution)**2
pdf_probas.sum()  # 0.999999...

plt.figure(figsize=(8, 4))
plot_gaussian_mixture(gm, X)
save_fig("gaussian_mixtures_diagram")
plt.show()

# Various covariance matrices specifications
gm_full = GaussianMixture(
    n_components=3, n_init=10, covariance_type="full", random_state=42)
gm_tied = GaussianMixture(
    n_components=3, n_init=10, covariance_type="tied", random_state=42)
gm_spherical = GaussianMixture(
    n_components=3, n_init=10, covariance_type="spherical", random_state=42)
gm_diag = GaussianMixture(
    n_components=3, n_init=10, covariance_type="diag", random_state=42)
gm_full.fit(X)
gm_tied.fit(X)
gm_spherical.fit(X)
gm_diag.fit(X)


compare_gaussian_mixture(gm_tied, gm_spherical, X)
save_fig("covariance_type_diagram")
plt.show()
