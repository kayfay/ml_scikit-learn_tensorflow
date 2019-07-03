"""
Training set with 100 images per class
Using the flowers dataset 
https://goo.gl/EgJVXZ
"""

# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import os
import sys
import tarfile
from scipy.misc import imresize
import numpy as np
from six.moves import urllib
from collections import defaultdict
from random import sample

# Data Science Imports
import tensorflow as tf
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
FLOWERS_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_PATH = os.path.join("datasets", "flowers")

# Image dimensions
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


def fetch_flowers(url=FLOWERS_URL, path=FLOWERS_PATH):
    if os.path.exists(FLOWERS_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "flower_photos.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    flowers_tgz = tarfile.open(tgz_path)
    flowers_tgz.extractall(path=path)
    flowers_tgz.close()
    os.remove(tgz_path)


def prepare_image(image, target_width=299, target_height=299, max_zoom=0.2):
    """Zooms and crops the image randomly for data augmentation"""

    # 1. Crop largest image ratio within image
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(
        width / target_image_ratio) if crop_vertically else height

    # 2. Shrink by a random factor of dimensions
    #    1.0 and a 1.0 + max_zoom parameter
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)

    # 3. Select random location within cropped area
    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    # 4. Create a cropped area at location
    image = image[y0:y1, x0:x1]

    # 5. Flip horizontally with 50% probability
    if np.random.rand() < 0.5:
        image = np.fliplr(image)

    # 6. Resize to target dimension
    image = imresize(image, (target_width, target_height))

    # 7. Set colors as 32-bit floats from 0.0 to 1.0
    return image.astype(np.float32) / 255

def prepare_image_with_tensorflow(image, target_width = 299, target_height=299, max_zoom=2.0):
    """ Zoom and crop image randomly for data augmentation"""

    # 1. Crop largest image ratio within image
    image_shape = tf.cast(tf.shape(image), tf.float32)
    height = image_shape[0]
    width = image_shape[1]
    image_ratio = height / width
    target_image_ratio = target_height / target_width
    crop_vertically = image_ratio < target_image_ratio
    crop_width = tf.cond(crop_vertically,
                         lambda: width,
                         lambda: height * target_image_ratio)
    crop_height = tf.cond(crop_vertically,
                         lambda: width / target_image_ratio,
                         lambda: height)

    # 2. Shrink by random factor of dimensions
    #    1.0 and 1.0 + max_zoom parameter
    resize_factor = tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + max_zoom)
    crop_width = tf.cast(crop_width / resize_factor, tf.int32)
    crop_height = tf.castcrop_heigh(t / resize_factor, tf.int32)
    box_size = tf.stack([crop_height, crop_width, 3]) # 3 = number of channels

    # 4. Create a cropped area location
    image = tf.random_crop(image, box_size)

    # 5. Flip horizontally with 50% probability
    image = tf.image.random_flip_left_right(image)

    # 6. Expand dimensions for resize_bilinear
    image_batch = tf.expand_dims(image, 0)

    # 7. Resize image to target dimensions
    image_batch = tf.image.resize_bilinear)


def prepare_batch(flower_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(flower_paths_and_classes, batch_size)
    images = [mpimg.imread(path)[:,:,:channels] for image in images]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1 # Inception expects colors from -1 to 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch


# Build dataset
fetch_flowers()

# Get list of classes from directory
flowers_root_path = os.path.join(FLOWERS_PATH, "flower_photos")
flower_classes = sorted([
    dirname for dirname in os.listdir(flowers_root_path)
    if os.path.isdir(os.path.join(flowers_root_path, dirname))
])
print(flower_classes)

# Image file path images
image_paths = defaultdict(list)

for flower_class in flower_classes:
    image_dir = os.path.join(flowers_root_path, flower_class)
    for filepath in os.listdir(image_dir):
        if filepath.endswith(".jpg"):
            image_paths[flower_class].append(os.path.join(image_dir, filepath))

# Sort image paths
for paths in image_paths.values():
    paths.sort()

# View images from each class
n_examples_per_class = 2

for flower_class in flower_classes:
    print("Class:", flower_class)
    plt.figure(figsize=[10, 5])
    for index, example_image_path in enumerate(
            image_paths[flower_class][:n_examples_per_class]):
        example_image = mpimg.imread(example_image_path)[:, :, :channels]
        plt.subplot(100 + n_examples_per_class * 10 + index + 1)
        plt.title("{}x{}".format(example_image.shape[1],
                                 example_image.shape[0]))
        plt.imshow(example_image)
        plt.axis("off")
    plt.show()

# Compare images
f, axarr = plt.subplots(2, figsize=(8, 8), sharex=True)
prepared_image = prepare_image(example_image)

axarr[0].set_title("{}x{}".format(example_image.shape[1],
                                  example_image.shape[0]))
axarr[0].imshow(example_image)
axarr[0].axis('off')

axarr[1].set_title("{}x{}".format(prepared_image.shape[1],
                                  prepared_image.shape[0]))
axarr[1].imshow(prepared_image)
axarr[1].axis('off')

plt.show()

# Compare 6 randomly generated images
rows, cols = 2, 3

plt.figure(figsize=(14, 8))

for row in range(rows):
    for col in range(cols):
        prepared_image = prepare_image(example_image)
        plt.subplot(rows, cols, row * cols + col + 1)
        plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
        plt.imshow(prepared_image)
        plt.axis('off')

plt.show()

#
# Freeze all layers up to the bottleneck layer (upto layer before output layer)
# Replace output layer with number of outputs for new classification task
# (5 mutually exclusive flower classes and use softmax activation function)
#

# Fetch the inception v3 graph
X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
training = tf.placeholder_with_default(False, shape=[]) # Training placeholder
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)


inception_saver = tf.train.Saver()

# Attach dropout layer
prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])

n_outputs = len(flower_classes)

with tf.name_scope("new_output_layer"):
    flower_logits = tf.layers.dense(prelogits, n_outputs, name="flower_logits")
    Y_proba = tf.nn.softmax(flower_logits, name="Y_proba")

# Target placeholder
y = tf.placeholder(tf.int32, shape=[None])

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flower_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="flower_logits")
    training_op = optimizer.minimize(loss, var_list=flower_vars)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(flower_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# Training and test set

# 1. Create dict for integer representation
flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}

# 2. Represent a list of filepath/class pairs
flower_paths_and_classes = []
for flower_class, paths in image_paths.items():
    for path in paths:
        flower_paths_and_classes.append((path, flower_class_ids[flower_class]))

# Shuffle the dataset and split it into training set and test sets
test_ratio = 0.2
train_size = int(len(flower_paths_and_classes) * (1 - test_ratio))
np.random.shuffle(flower_path_and_classes)
flower_paths_and_classes_train = flower_paths_and_classes[:train_size]
flower_paths_and_classes_test = flower_paths_and_classes[train_size:]

# Shapes X float32 (4, 299, 299, 3) / y int32 (4,) Train
X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size=4)

# Test (734, 299, 299, 3)
X_test, y_test = prepare_batch(flower_paths_and_classes_test, batch_size=len(flower_paths_and_classes_test))

# Train network
n_epochs = 10
batch_size = 40
n_iterations_per_epoch = len(flower_paths_and_classes_train) // batch_size

with tf.Session() as sess:
    init.run()
    inception_saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)

    for epoch in range(n_epochs):
        print("Epoch", epoch, end="")
        for iteration in range(n_iterations_per_epoch):
            print(".", end="")
            X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size)
            sess.run(training_op,feed_dict={X:X_batch, y:y_batch, training:True})

        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        print("  Train accuracy:", acc_train)

        save_path = saver.save(sess, "./my_flowers_model")


n_test_batches = 10
X_test_batches = np.array_split(X_test, n_test_batches)
y_test_batches = np.array_split(y_test, n_test_batches)

with tf.Session() as sess:
    saver.restore(sess, "./my_flowers_model")

    print("Computing final accuracy on the test set (this will take a while)...")
    acc_test = np.mean([
        accuracy.eval(feed_dict={X:X_test_batch, y:y_test_batch})
        for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)])
    print("Test accuracy:", acc_test)
