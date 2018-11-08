# Python 2 and 3
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np
import os

# Data Science Imports
import tensorflow as tf
from sklearn.datasets import load_sample_image

# Graph Imports
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12

# Config
PROJECT_ROOT_DIR = "."


# Declare Function
def save_fig(fig_id, tight_layout=True):
    if not os.path.exists('images'):
        os.makedirs('images')
    path = os.path.join(PROJECT_ROOT_DIR, 'images', fig_id, '.png')
    plt.savefig(path, format='png', dpi=300)



def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")


def plot_color_image(image):
    plt.imshow(image.astype(np.uint8), interpolation="nearest")
    plt.axis("off")


# Prepare data
china = load_sample_image('china.jpg')
flower = load_sample_image('flower.jpg')
image = china[150:220, 130:250]
height, width, channels = image.shape
image_grayscale = image.mean(axis=2).astype(np.float32)
images = image_grayscale.reshape(1, height, width, 1)

# Feature maps
fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
fmap[:, 3, 0, 0] = 1
fmap[3, :, 0, 1] = 1
plot_image(fmap[:, :, 0, 0])
plt.show()
plot_image(fmap[:, :, 0, 1])
plt.show()

# Create placeholder nodes and graph nodes
X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
feature_maps = tf.constant(fmap)
convolution = tf.nn.conv2d(
    X, feature_maps, strides=[1, 1, 1, 1], padding="SAME")

# Session output
with tf.Session() as sess:
    output = convolution.eval(feed_dict={X: images})

plot_image(images[0, :, :, 0])
save_fig("china_original", tight_layout=False)
plt.show()
print("inital image")

plot_image(output[0, :, :, 0])
save_fig("china_vertical", tight_layout=False)
plt.show()
print("image with vertical features")

plot_image(output[0, :, :, 1])
save_fig("china_horizontal", tight_layout=False)
plt.show()
print("image with horizontal features")
