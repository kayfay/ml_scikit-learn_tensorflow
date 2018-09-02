# Classifiy flower type based on petal widths
# Predictions done with logistic regression
# Using softmax regression build a contour plot

# Common Imports
import os
import numpy as np

# ML Imports
from sklearn.linear_model import LogisticRegression

# Dataset imports
from sklearn import datasets

# Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

# Instantiate softmax classifier
softmax_reg = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)

x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500).reshape(-1, 1),
    np.linspace(0, 3.5, 200).reshape(-1, 1),
)

X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")
custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
save_fig("softmax_regression_contour_plot")
plt.show()

# Prediction and prediction probabilities
print("Prediction: ", softmax_reg.predict([[5, 2]]))
print("Probabilities:" , softmax_reg.predict_proba([[5, 2]]))
