# Python 2 and 3 compatability
from __future__ import division, unicode_literals, print_function

# Common Imports
import os
import numpy as np

# ML Imports
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn import datasets

# Graph Imports
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# Declare Functions
def save_fig(fig_id, tight_layout=True):
    if not os.path.exists('images'):
        os.makedirs('images')
    path = os.path.join(PROJECT_ROOT_DIR, 'images', fig_id + '.png')
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # w0*x0 + w1*x1 + b = 0
    # x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=-2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


# Directory settings
PROJECT_ROOT_DIR = '.'


# Declare classes
class MyLinearSVC(BaseEstimator):
    def __init__(self,
                 C=1,
                 eta0=1,
                 eta_d=10000,
                 n_epochs=1000,
                 random_state=None):
        self.C = C
        self.eta0 = eta0
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.eta_d = eta_d

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)

    def fit(self, X, y):
        # Random Init
        if self.random_state:
            np.random.seed(self.random_state)
        w = np.random.randn(X.shape[1], 1)  # n feat weights
        b = 0

        # m = len(X)
        t = y * 2 - 1  # -1 if t==0, +1 if t ==1
        X_t = X * t
        self.Js = []

        # Training
        for epoch in range(self.n_epochs):
            support_vectors_idx = (X_t.dot(w) + t * b < 1).ravel()
            X_t_sv = X_t[support_vectors_idx]
            t_sv = t[support_vectors_idx]

            J = 1 / 2 * np.sum(w * w) + self.C * (
                np.sum(1 - X_t_sv.dot(w)) - b * np.sum(t_sv))
            self.Js.append(J)

            w_gradient_vector = w - self.C * np.sum(
                X_t_sv, axis=0).reshape(-1, 1)
            b_derivative = -C * np.sum(t_sv)

            w = w - self.eta(epoch) * w_gradient_vector
            b = b - self.eta(epoch) * b_derivative

        self.intercept_ = np.array([b])
        self.coef_ = np.array([w])
        support_vectors_idx = (X_t.dot(w) + t * b < 1).ravel()
        self.support_vectors_ = X[support_vectors_idx]
        return self

    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_[0]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)


# Create Datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # (petal lenght.cm, petal width.cm)
y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1)  # Iris-Verginica

# Fit model
C = 2
svm_clf = MyLinearSVC(
    C=C, eta0=10, eta_d=1000, n_epochs=60000, random_state=42)
svm_clf.fit(X, y)
# svm_clf.predict(np.array([[5, 2], 4, 1]))

# Plot learning curve
plt.plot(range(svm_clf.n_epochs), svm_clf.Js)
plt.axis([0, svm_clf.n_epochs, 0, 100])
save_fig('l-SVC_BGD_learning_curve')
plt.show()

# Intercept and Coefficients
print(svm_clf.intercept_, svm_clf.coef_)

# Compare to sk_base
svm_clf2 = SVC(kernel="linear", C=C)
svm_clf2.fit(X, y.ravel())
print(svm_clf2.intercept_, svm_clf2.coef_)

# Plot SVMs
yr = y.ravel()
plt.figure(figsize=(12, 3.2))
plt.subplot(121)
plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^", label="Iris-Virginica")
plt.plot(
    X[:, 0][yr == 0], X[:, 1][yr == 0], "bs", label="Not Iris-Virginica")
plot_svc_decision_boundary(svm_clf, 4, 6)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.title("MyLinearSVC", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])

plt.subplot(122)
plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")
plot_svc_decision_boundary(svm_clf2, 4, 6)
plt.xlabel("Petal length", fontsize=14)
plt.title("SVC", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])
save_fig("my_svc_comparison")
plt.show()
