"""

CNN
    Using a stride of 1 in convolutional layer 1
    Using a dropout/normalization layer with 25% after layer 2
    Using a dropout/normalization layer with 50% after fully connected layer
    Using early stopping in training

"""
# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np

# Data Science Imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Graph Imports
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Config
PROJECT_ROOT_DIR = "."

# Hyperparameters
# Node input hyperparameters
height = 28
width = 28
channels = 1
n_inputs = height * width

# Convolutional layer 1 hyperparameters
conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

# Convolutional layer 2 hyperparameters
conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"
conv2_dropout_rate = 0.25

# Pooling layer hyperparameters
pool3_fmaps = conv2_fmaps

# Fully connected normalization, dropout layer
n_fc1 = 128
fc1_dropout_rate = 0.5

# Output layer
n_outputs = 10

# Training hyperparameters
n_epochs = 1000
batch_size = 50

# Model hyperparameters
best_loss_val = np.infty
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None


# Declare Functions

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


def get_model_params():
    """
    Get models state, variables values
    """
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {
        gvar.op.name: value
        for gvar, value in zip(gvars,
                               tf.get_default_session().run(gvars))
    }


def restore_model_params(model_params):
    """
    Restores previous state, speeds up early stopping by saving to memory
    """
    gvar_names = list(model_params.keys())
    assign_ops = {
        gvar_name:
        tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
        for gvar_name in gvar_names
    }
    init_values = {
        gvar_name: assign_op.inputs[1]
        for gvar_name, assign_op in assign_ops.items()
    }
    feed_dict = {
        init_values[gvar_name]: model_params[gvar_name]
        for gvar_name in gvar_names
    }
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


# Load Data
mnist = input_data.read_data_sets("/tmp/data/")

# Convolutoinal Neural Network
with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')

conv1 = tf.layers.conv2d(
    X_reshaped,
    filters=conv1_fmaps,
    kernel_size=conv1_ksize,
    strides=conv1_stride,
    padding=conv1_pad,
    activation=tf.nn.relu,
    name="conv1")
conv2 = tf.layers.conv2d(
    conv1,
    filters=conv2_fmaps,
    kernel_size=conv2_ksize,
    strides=conv2_stride,
    padding=conv2_pad,
    activation=tf.nn.relu,
    name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(
        conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 14 * 14])
    pool3_flat_drop = tf.layers.dropout(
        pool3_flat, conv2_dropout_rate, training=training)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(
        pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# Training model
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(
                training_op,
                feed_dict={
                    X: X_batch,
                    y: y_batch,
                    training: True
                })

            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={
                    X: mnist.validation.images,
                    y: mnist.validation.labels
                })

                # Every 100 iterations evaluate model on validation set
                # For 100 evaluations in row with no accuracy progress
                #  interrupt training

                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()

                else:
                    checks_since_last_progress += 1

            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_batch, y: y_batch})

            print(
                "Epoch {}, train accuracy: {:.4f}%, valid, accuracy: {:.4f}%, valid, best loss: {:.6f}".
                format(epoch, acc_train * 100, acc_val * 100, best_loss_val))

            # Early stopping
            if checks_since_last_progress > max_checks_without_progress:
                print("Early stopping!")
                break

    # Restore best model after training
    # Save model to ram when outperforming previously saved
    if best_model_params:
        restore_model_params(best_model_params)

    acc_test = accuracy.eval(feed_dict={
        X: mnist.test.images,
        y: mnist.test.labels
    })

    print("Final accuracy on test set:", acc_test)
    save_path = saver.save(sess, "./my_mnist_model")
