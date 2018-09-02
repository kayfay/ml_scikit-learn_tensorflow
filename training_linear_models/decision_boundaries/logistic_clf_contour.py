# Classifiy flower type based on petal widths
# Predictions done with logistic regression
# Contour plot

# Common Imports
import os
import numpy as np

# ML Imports
from sklearn.linear_model import LogisticRegression

# Dataset imports
from sklearn import datasets

# Matplotlib imports
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Declare directory variables
PROJECT_ROOT_DIR = "."
IMG_DIR = PROJECT_ROOT_DIR + "/images"

# Declare functions


def save_fig(fig_id, tight_layout=True):
    if not os.path.isdir(IMG_DIR):
        os.makedirs(IMG_DIR)
    path = os.path.join(IMG_DIR, fig_id + ".png")
    print("Saving figure", fig_id)
    plt.savefig(path, format='png', dpi=300)


# Initalize variables
iris = datasets.load_iris()
X = iris['data'][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # create 1 or 0

# Create visualization of decision plot
X = iris["data"][:, (2, 3)]  # petal legnth/width
y = (iris["target"] == 2).astype(np.int)

# Instantiate classifier
log_reg = LogisticRegression(C=10**10)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
    np.linspace(2.9, 7, 500).reshape(-1, 1),
    np.linspace(0.8, 2.7, 200).reshape(-1, 1),
)

X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 0, 0], X[y == 0, 1], "bs")
plt.plot(X[y == 1, 0], X[y == 1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right +
             log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="b", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
save_fig("logistic_regression_contour_plot")
plt.show()
