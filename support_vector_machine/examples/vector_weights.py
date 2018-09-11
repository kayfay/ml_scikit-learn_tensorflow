# Python 2 and 3 support
from __future__ import print_function, division, unicode_literals

# Common Imports
import os
import numpy as np

# Graph Imports
import matplotlib.pyplot as plt

# Config settings
PROJECT_ROOT_DIR = '.'


# Common functions
def save_fig(fig_id, tight_layout=True):
    if not os.path.exists('images'):
        os.mkdirs('images')
    path = os.path.join(PROJECT_ROOT_DIR, 'images', fig_id + '.png')
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=r'png', dpi=300)


def plot_2D_decision_function(w, b, ylabel=True, x1_lim=[-3, 3]):
    x1 = np.linspace(x1_lim[0], x1_lim[1], 200)
    y = w * x1 + b
    m = 1 / w

    plt.plot(x1, y)
    plt.plot(x1_lim, [1, 1], "k:")
    plt.plot(x1_lim, [-1, -1], "k:")
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot([m, m], [0, 1], "k--")
    plt.plot([-m, -m], [0, -1], "k--")
    plt.plot([-m, m], [0, 0], "k-o", linewidth=3)
    plt.axis(x1_lim + [-2, 2])
    plt.xlabel(r"$x_1$", fontsize=16)
    if ylabel:
        plt.ylabel(r"$w_1 x_1$", rotation=0, fontsize=16)
    plt.title(r"$w_1 = {}$".format(w), fontsize=16)


plt.figure(figsize=(12, 3.2))
plt.subplot(121)
plot_2D_decision_function(1, 0)
plt.subplot(122)
plot_2D_decision_function(0.5, 0, ylabel=False)
save_fig("small_w_large_margin_plot")
plt.show()
