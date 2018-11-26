"""
CNN Image classification with:

GoogLeNet architecture http://goo.gl/tCFzVs
Inception Module
    input
    convolution layer x2 & max pooling layer
    convolution x4 various kernels
    depth layer / concatenation layer
"""

# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import re
import os
import sys
import tarfile
from six.moves import urllib
from PIL import Image

# Data Science Imports
import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

# Graph Imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Config
PROJECT_ROOT_DIR = "."
IMAGES_DIR = os.path.join("images", "cnn")
TF_MODELS_URL = 'http://download.tensorflow.org/models'
INCEPTION_V3_URL = TF_MODELS_URL + "/inception_v3_2016_08_28.tar.gz"
INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH,
                                            "inception_v3.ckpt")
CLASS_NAME_REGEX = re.compile(r"^n\d+\s+(.*)\s*$", re.M | re.U)

# Hyperparameters

# Image hyperparameters
width = 299
height = 299
channels = 3


# Declare Functions
def save_fig(fig_id, tight_layout=True):
    if not os.path.exists('images'):
        os.makedirs('images')
    path = os.path.join(PROJECT_ROOT_DIR, 'images', fig_id, ".png")
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


def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()


def fetch_pretrained_inception_v3(url=INCEPTION_V3_URL, path=INCEPTION_PATH):
    if os.path.exists(INCEPTION_V3_CHECKPOINT_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "inception_v3.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    inception_tgz = tarfile.open(tgz_path)
    inception_tgz.extractall(path=path)
    inception_tgz.close()
    os.remove(tgz_path)


def load_class_names():
    with open(
            os.path.join("datasets", "inception", "imagenet_class_names.txt"),
            "rb") as f:
        content = f.read().decode("utf-8")
        return CLASS_NAME_REGEX.findall(content)


def prep_image(image, height=height, width=width, channels=channels):
    img = Image.fromarray(image)
    return np.array(img.resize((width, height)))


def load_images(folder=IMAGES_DIR):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))[:, :, :channels]
        if img is not None:
            images.append(prep_image(img))
    return images


# Convolutional Neural Network

# Download previous model checkpoint
fetch_pretrained_inception_v3()

# Create Inception v3 model
X = tf.placeholder(tf.float32, shape=[None, width, height, channels], name="X")
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(
        X, num_classes=1001, is_training=False)

predictions = end_points["Predictions"]
saver = tf.train.Saver()

# Classify the images from
# http://www.sciencekids.co.nz/images/pictures/animals96/
X_test = load_images()[2].reshape(-1, height, width, channels)

# Restore previous model checkpoint
with tf.Session() as sess:
    saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
    predictions_val = predictions.eval(feed_dict={X: X_test})

class_names = ["background"] + load_class_names()

# Show prediction image
plt.imshow(X_test[0])
plt.show()

# Prediction names
most_likely_class_index = np.argmax(predictions_val[0])
top = class_names[most_likely_class_index]
top_5 = np.argpartition(predictions_val[0], -5)[-5:]
top_5 = reversed(top_5[np.argsort(predictions_val[0][top_5])])

# Display predictions
for i in top_5:
    print("{0}: {1:.2f}%".format(class_names[i], 100 * predictions_val[0][i]))
