"""
Logistic Regression with Mini-Batch Gradient Descent using Tensorflow
Using the Moons dataset
Defines a graph
Uses Saver to saves checkpoints during training and the final model
Restore from last checkppoint if interrupted
Uses name scopes
Uses summaries to visualize the learning curves in tensorboard
"""

# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import os
import numpy as np
from datetime import datetime

# Data science Imports
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.metrics import precision_score, recall_score

# Graph Imports
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Config


# Declare Functions
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def random_batch(X_train, y_train, batch_size):
    """Pick random instances from 1st and 2nd arguments"""
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


def logistic_regression(X, y, initializer=None, seed=42, learning_rate=0.01):
    n_inputs_including_bias = int(X.get_shape()[1])
    with tf.name_scope("logistic_regression"):
        with tf.name_scope("model"):
            if initializer is None:
                initializer = tf.random_uniform(
                    [n_inputs_including_bias, 1], -1.0, 1.0, seed=seed)
            theta = tf.Variable(initializer, name="theta")
            logits = tf.matmul(X, theta, name="logits")
            y_proba = tf.sigmoid(logits)
        with tf.name_scope("train"):
            loss = tf.losses.log_loss(y, y_proba, scope="loss")
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar('log_loss', loss)
        with tf.name_scope("init"):
            init = tf.global_variables_initializer()
        with tf.name_scope("save"):
            saver = tf.train.Saver()
    return y_proba, loss, training_op, loss_summary, init, saver


# Create a dataset
m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

# Plot the dataset
plt.plot(
    X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
plt.plot(
    X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")

plt.legend()
plt.show()

# Add an extra bias feature to every instance
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]

# Reshape y_train into a column vector
y_moons_column_vector = y_moons.reshape(-1, 1)

# Split training and test set
test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]

# Reset default graph
reset_graph()

# Add features x_1^2, X_2^2, X_1^3, X_2^3 by hand
X_train_enhanced = np.c_[X_train,
                         np.square(X_train[:, 1]),
                         np.square(X_train[:, 2]), X_train[:, 1]**3,
                         X_train[:, 2]**3]

X_test_enhanced = np.c_[X_test,
                        np.square(X_test[:, 1]),
                        np.square(X_test[:, 2]), X_test[:, 1]**3, X_test[:, 2]
                        **3]

# Create graph
# Define inputs and log directory
n_inputs = 2 + 4  # the original two and the 4 poly feats
logdir = log_dir("logreg")

# Define placeholders
X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

# Define nodes for logistic regression and the model saver
y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(
    X, y)

# Write to the default graph
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Train classifier
n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "/tmp/tf_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./tf_logreg_model"

with tf.Session() as sess:
    if os.path.isfile(
            checkpoint_epoch_path):  # Restart from chkpt if interrupted
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted, Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train_enhanced, y_train,
                                            batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, summary_str = sess.run(
            [loss, loss_summary], feed_dict={
                X: X_test_enhanced,
                y: y_test
            })
        file_writer.add_summary(summary_str, epoch)
        if epoch % 500 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, 'wb') as f:
                f.write(b"%d" % (epoch + 1))

    saver.save(sess, final_model_path)
    y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
    os.remove(checkpoint_epoch_path)

# Make predictions based on 50% estimated probabilities
y_pred = (y_proba_val >= 0.5)

print("Pression: ", precision_score(y_test, y_pred))
print("Recall Scores: ", recall_score(y_test, y_pred))

# Plot Presission / Recall Scores
y_pred_idx = y_pred.reshape(-1)
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label="Positive")
plt.plot(
    X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label="Negative")
plt.legend()
plt.show()
