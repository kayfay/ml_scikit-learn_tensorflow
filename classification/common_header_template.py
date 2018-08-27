# To support both python2 & 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# Set seed value
np.random.seed(42)

# matplotlib imports
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Save fig directory
PROJECT_ROOT_DIR = "."
DIRECTORY = "classification"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", DIRECTORY, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

