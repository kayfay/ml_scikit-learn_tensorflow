# Classifiy flower type based on petal widths
# Predictions done with logistic regression
# Decision plot

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

# Instantiate classifier
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Create X/Y axis values and boundaries
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

# Generate plots

# Create a plot for (X, y(1|0) Decision boundary
plt.figure(figsize=(8, 3))
plt.plot(X[y == 0], y[y == 0], "bs")
plt.plot(X[y == 1], y[y == 1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.text(
    decision_boundary + 0.02,
    0.15,
    "Decision  boundary",
    fontsize=14,
    color="k",
    ha="center")
plt.arrow(
    decision_boundary,
    0.08,
    -0.3,
    0,
    head_width=0.05,
    head_length=0.1,
    fc="b",
    ec='b')
plt.arrow(
    decision_boundary,
    0.92,
    0.3,
    0,
    head_width=0.05,
    head_length=0.1,
    fc='g',
    ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
save_fig("logistic_regression_plot")
plt.show()

# Exploration
print(iris.DESCR, "Summary Statistics: \n", "Keys: ", list(iris.keys()),
      "\n\n", "Decision boundary: ",
      decision_boundary)

# Show prediction
print("\nPredict for 1.7cm, 1.5cm width/length: ",
      log_reg.predict([[1.7], [1.5]]), "\n")



