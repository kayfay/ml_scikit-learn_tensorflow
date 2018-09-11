# Python 2 and 3
from __future__ import division, unicode_literals, print_function

# Common Imports
import os

# Timer Import
import time

# ML Imports
from sklearn.svm import SVC
from sklearn.datasets import make_moons

# Graph Imports
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Directory Config
PROJECT_DIRECTORY = '.'


# Declare Functions
def save_fig(fig_id, tight_layout=True):
    if not os.path.isdir('images'):
        os.makedir('images')
    path = os.path.join(PROJECT_DIRECTORY, 'images', fig_id + '.png')
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=r"png", dpi=300)


# Instantiate / modify dataset
X, y = make_moons(n_samples=1000, noise=0.4, random_state=42)
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
save_fig('noise_moon')
plt.show()

# Test training time based on features
tol = 0.1
tols = []
times = []

for i in range(10):
    svm_clf = SVC(kernel="poly", gamma=3, C=10, tol=tol, verbose=1)
    t1 = time.time()
    svm_clf.fit(X, y)
    t2 = time.time()
    times.append(t2 - t1)
    tols.append(tol)
    print(i, tol, t2 - t1)
    tol /= 10

plt.semilogx(tols, times)
save_fig('timed_SVC')
plt.show()
