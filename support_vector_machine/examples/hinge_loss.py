# Python 2 and 3 support
from __future__ import division, unicode_literals, print_function

# Common Imports
import os
import numpy as np

# Graph Imports & Config
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# ML Imports

# Directory Config
PROJECT_ROOT_DIR = '.'

# Declare Functions


def save_fig(fig_id, tight_layout=True):
    if not os.path.isdir('images'):
        os.makedirs('images')
    path = os.path.join(PROJECT_ROOT_DIR, 'images', fig_id + '.png')
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=r'png', dpi=300)


# Generate t space and values
t = np.linspace(-2, 4, 200)

# Hinge loss function & graph
h = np.where(1 - t < 0, 0, 1 - t)  # max(0, 1-t)

plt.figure(figsize=(5, 2.8))
plt.plot(t, h, "b-", linewidth=2, label="$max(0, 1-t)$")
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.yticks(np.arange(-1, 2.5, 1))
plt.xlabel("$t$", fontsize=16)
plt.axis([-2, 4, -1, 2.5])
plt.legend(loc="upper right", fontsize=16)
save_fig("hinge_plot")
plt.show()
