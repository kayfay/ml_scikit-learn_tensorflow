# A 3D plot of the decision boundary
# suppor vector margins and decision function
# plotted over x and y axis of the dataset features

# Support for Python 2 and 3
from __future__ import print_function, unicode_literals, division

# Common Imports
import os
import numpy as np

# Dataset import
from sklearn import datasets

# ML Imports
from sklearn.svm import LinearSVC

# Graph Imports
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Save directory
PROJECT_ROOT_DIR = "."

# Declare functions


def save_fig(fig_id, tight_layout=True):
    if not os.path.isdir('images'):
        os.mkdir(PROJECT_ROOT_DIR + '/images')
    path = os.path.join(PROJECT_ROOT_DIR, 'images', fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=r'png', dpi=300)


def plot_3D_decision_function(ax, w, b, x1_lim=[4, 6], x2_lim=[0.8, 2.8]):
    x1_in_bounds = (X[:, 0] > x1_lim[0]) & (X[:, 0] < x1_lim[1])
    X_crop = X[x1_in_bounds]
    y_crop = y[x1_in_bounds]
    x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
    x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
    x1, x2 = np.meshgrid(x1s, x2s)
    xs = np.c_[x1.ravel(), x2.ravel()]
    df = (xs.dot(w) + b).reshape(x1.shape)
    # m = 1 / np.linalg.norm(w)
    boundary_x2s = -x1s * (w[0] / w[1]) - b / w[1]
    margin_x2s_1 = -x1s * (w[0] / w[1]) - (b - 1) / w[1]
    margin_x2s_2 = -x1s * (w[0] / w[1]) - (b + 1) / w[1]
    ax.plot_surface(
        x1s,
        x2,
        np.zeros_like(x1),
        color="b",
        alpha=0.2,
        cstride=100,
        rstride=100)
    ax.plot(x1s, boundary_x2s, 0, "k-", linewidth=2, label=r"$h=0$")
    ax.plot(x1s, margin_x2s_1, 0, "k--", linewidth=2, label=r"$h=\pm 1$")
    ax.plot(x1s, margin_x2s_2, 0, "k--", linewidth=2)
    ax.plot(X_crop[:, 0][y_crop == 1], X_crop[:, 1][y_crop == 1], 0, "g^")
    ax.plot_wireframe(x1, x2, df, alpha=0.3, color="k")
    ax.plot(X_crop[:, 0][y_crop == 0], X_crop[:, 1][y_crop == 0], 0, "bs")
    ax.axis(x1_lim + x2_lim)
    ax.text(4.5, 2.5, 3.8, "Decision function $h$", fontsize=15)
    ax.set_xlabel(r"Petal length", fontsize=15)
    ax.set_ylabel(r"Petal width", fontsize=15)
    ax.set_zlabel(r"$h = \mathbf{w}^T \mathbf{x} + b$", fontsize=18)
    ax.legend(loc="upper left", fontsize=16)


# Data
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64)

svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)
svm_clf2.fit(X, y)

# Plots
fig = plt.figure(figsize=(11, 6))
ax1 = fig.add_subplot(111, projection='3d')
plot_3D_decision_function(ax1, w=svm_clf2.coef_[0], b=svm_clf2.intercept_[0])

save_fig("iris_3D_plot")
plt.show()
