# Show examples of how feature importances on an image of a number
# color mapped by means of classification using rnd forest clf
# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import os

# ML Imports
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier

# Graph Imports
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Directory Config
PROJECT_ROOT_DIR = '.'


# Declare Functions
def image_path(fig_id):
    if not os.path.exists('images'):
        os.makedirs('images')
    return os.path.join(PROJECT_ROOT_DIR, 'images', fig_id)


def save_fig(fig_id, tight_layout=True):
    print("Saving ", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.hot, interpolation="nearest")
    plt.axis("off")


mnist = fetch_mldata('MNIST original')
rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(mnist['data'], mnist['target'])

plot_digit(rnd_clf.feature_importances_)

cbar = plt.colorbar(ticks=[
    rnd_clf.feature_importances_.min(),
    rnd_clf.feature_importances_.max()
])

cbar.ax.set_yticklabels(['Not important', 'Very important'])

save_fig("mnist_feature_importance_plot")
plt.show()
