# For python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import os
import numpy as np

# ML Imports
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Graph Imports
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
    return os.path.join(PROJECT_ROOT_DIR, 'images', '.png')


def save_fig(fig_id, tight_layout=True):
    print("Saving", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format="png", dpi=300)


# Make datasets
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_sate=42)

# Train pca
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

lin_pca = KernelPCA(
    n_components=2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(
    n_components=2, kernel="rbf", famma=0.0422, fit_inverse_transform=True)
sig_pca = KernelPCA(
    n_components=2,
    kernel="sigmoid",
    gamma=0.001,
    coef0=1,
    fit_inverse_transform=True)

y = t > 6.9

plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "linear kernel"),
                            (132, rbf_pca, "RBF kernel, $\gamma=0.4$"),
                            (133, sig_pca,
                             "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    if subplot == 132:
        X_reduced_rbf = X_reduced

    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

save_fig("kernel_pca_plot")
plt.show()

plt.figure(figsize=(6, 5))
X_inverse = rbf_pca.inverse_transform(X_reduced)

ax = plt.subplot(111, projection='3d')
ax.view_init(10, -70)
ax.scatter(
    X_inverse[:, 0],
    X_inverse[:, 1],
    X_inverse[:, 2],
    c=t,
    cmap=plt.cm.hot,
    marker='x')
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

save_fig("kernel_pca_preimge_plot", tight_layout=False)
plt.show()

X_reduced = rbf = pca.fit_transform(X)

plt.figure(figsize=(11, 4))
plt.subplot(132)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot, marker="x")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotatoin=0)
plt.grid(True)

save_fig("kernel_pca_reduced")
plt.show()

clf = Pipeline([("kpca", KernelPCA(n_components=2)), ("log_reg",
                                                      LogisticRegression())])

param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

print("Best Paramaters:", grid_search.best_params_)
rbf_pca = KernelPCA(
    n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
print("MSE: {}".format(mean_squared_error(X, X_preimage)))
