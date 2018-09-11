# Support for py 2&3
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np
import matplotlib.pylab as plt
import os

# ML Imports
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC

# Config
PROJECT_ROOT_DIR = "."

# Graphing config
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# Declare functions
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


def save_fig(fig_id, tight_layout=True):
    if not os.path.exists("images"):
        os.makedirs("images")
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


# Import datasets
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# Pipeline for polynomial feature transform, scaling, and classifier
polynomial_svm_clf = Pipeline([("poly_features", PolynomialFeatures(degree=3)),
                               ("scaler",
                                StandardScaler()), ("svm_clf",
                                                    LinearSVC(
                                                        C=10,
                                                        loss="hinge",
                                                        random_state=42))])

# Pipeline for SVC tuning parameters
# d = 3, r = 1, C = 5, d = 10, r = 100, C = 5
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel='poly', degree=3, coef0=1, C=5))
])

poly_kernel_svm_clf.fit(X, y)

poly100_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
])

poly100_kernel_svm_clf.fit(X, y)

# Classifier transform  fitting model
polynomial_svm_clf.fit(X, y)

# Plot
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
save_fig("plot_dataset")
plt.show()

# Plot predictions
plot_predictions(polynomial_svm_clf, [-1.4, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

# Example plot
save_fig("polynomial_svc_plot")
plt.show()

# Plot grid tuned models
plt.figure(figsize=(11, 4))

plt.subplot(121)
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.subplot(122)
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=100, C=5$", fontsize=18)

save_fig("moons_kernelized_polynomial_svc_plot")
plt.show()
