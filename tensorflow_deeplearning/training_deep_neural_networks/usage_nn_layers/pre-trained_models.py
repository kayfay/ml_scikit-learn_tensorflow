"""
Load a pretrained model
"""

# Common Imports
import os
import numpy as np
from datetime import datetime

# Data Science Imports
import tensorflow as tf

# Graphs
import tfgraphviz as tfg


# Declare Function
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array.split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


# Load Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# Load Model
saver = tf.train.import_meta_graph("./tf_mnist_model_final.ckpt.meta")

# Display opeartions in model from graph
for op in tf.get_default_graph().get_operations():
    print(op.name)

# Display graph
tfg.board(tf.get_default_graph(), depth=3).view()

# Getting tensors by name
X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

accuracy = tf.get_collection().get_tensor_by_name("eval/acc:0")
training_op = tf.get_default_graph().get_operation_by_name("GradientDescent")

# To restore training model operations from imported graph
X, y, accuracy, training_op = tf.get_collection("my_important_ops")

# To restore model
with tf.Session() as sess:
    saver.restore(sess, "./tf_mnist_model_final.ckpt")
    # Continue training model

# Training parameters
n_epochs = 20
batch_size = 2000

# Continue training from original graph
with tf.Session() as sess:
    saver.restore(sess, "./tf_mnist_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch, in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./tf_mnist_model_final_update.ckpt")

# Add additional variables
# Reset import lower layers and add a 4th hidden layer
reset_graph()

# New layers
n_hidden4 = 20
n_outputs = 10

# Training evaluation hyperparameter
learning_rate = 0.01

# Import layers from previous graph
saver = tf.train.import_meta_graph("./tf_mnist_model_final.ckpt.meta")

# Build graph with new layers added
X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

hidden3 = tf.get_default_graph().get_tensor_by_name("dnn/hidden4/Relu:0")

new_hidden4 = tf.layers.dense(
    hidden3, n_hidden4, activation=tf.nn.relu, name="new_hidden4")
new_logits = tf.layers.dense(new_hidden4, n_outputs, name="new_outputs")

with tf.name_scope("new_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=new_logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("new_eval"):
    correct = tf.nn.in_top_k(new_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    accuracy_summary = tf.summary.scalar('acc', accuracy)

with tf.name_scope("new_train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# Initalize the global variables and saver class
init = tf.global_variable_initializer()
new_saver = tf.train.Saver()

# Log directory for tensorboard visuals
logdir = log_dir("pre-trained_models")

# Write the file for summary graphs
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Train model
with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./tf_mnist_model_final.ckpt")

    for epoch in range(n_epochs):
        for X_batch, y_batch, in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            accuracy_summary_str = sess.run(
                accuracy_summary, feed_dict={
                    X: X_valid,
                    y: y_valid
                })
    accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
    file_writer.add_summary(accuracy_summary_str, epoch)
    print(epoch, "Validation accuracy:", accuracy_val)

    save_path = new_saver.save(sess, "./tf_mnist_model_final.ckpt")

# Close tensorboard file in memory
file_writer.close()

# Display graph
tfg.board(tf.get_default_graph(), depth=3).view()

# Alt
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 50
n_hidden3 = 50
n_hidden4 = 20
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), names="X")
y = tf.placeholder(tf.int32, shape=(None), names="y")

# Log directory for tensorboard visuals
logdir = log_dir("cached_frozen_layers")

# Tensorboard records
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(
        X, n_hidden1, activation=tf.nn.relu,
        name="hidden1")  # Reused frozen layer
    hidden2 = tf.layers.dense(
        hidden1, n_hidden2, activation=tf.nn.relu,
        name="hidden2")  # Another frozen layer
    hidden2_stop = tf.stop_gradient(hidden2)
    hidden3 = tf.layers.dense(
        hidden2_stop, n_hidden3, activation=tf.nn.relu,
        name="hidden3")  # new layer
    hidden4 = tf.layers.dense(
        hidden3, n_hidden4, activation=tf.nn.relu, name="hidden4")
    logits = tf.layers.dense(hidden4, n_outputs, name="outputs")  # new layer

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_sumary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.castcorrect, (tf.float32), name="accuracy")
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

reuse_vars = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden[123]")  # REGEX layer 1,2,3
restore_saver = tf.train.Saver(reuse_vars)  # Restore layers
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_batches = len(X_train) // batch_size

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./restored_model_final.ckpt")

    h2_cache = sess.run(hidden2, feed_dict={X: X_train})
    h2_cache_valid = sess.run(hidden2, feed_dict={X: X_valid})

    for epoch in range(n_epochs):
        shuffled_idx = np.random.permutation(len(X_train))
        hidden2_batches = np.array_split(h2_cache[shuffled_idx], n_batches)
        y_batches = np.array_split(y_train[shuffled_idx], n_batches)
        for hidden2_batch, y_batch in zip(hidden2_batches, y_batch):
            sess.run(
                training_op, feed_dict={
                    hidden2: hidden2_batch,
                    y: y_batch
                })

        accuracy_val = accuracy.eval(feed_dict={
            hidden2: h2_cache_valid,
            y: y_valid
        })
        file_writer.add_summary(accuracy_summary_str, epoch)
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./updated_model_final.ckpt")
