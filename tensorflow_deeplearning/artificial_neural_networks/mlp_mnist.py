"""
Train a deep multiplayer perceptron on the MNIST image dataset for 98% accuracy as the goal
Save checkpoints
Restore checkpoits incase of interruption
Log summaries
Plot learning curves using tensorboard
"""
# Common Imports
import os
import numpy as np
from datetime import datetime

# Data Science Imports
import tensorflow as tf

# Graph Imports
import tfgraphviz as tfg


# Declare Functions
def log_dir(prefix=""):
    """Tensorboard logs dir"""
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# Get Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# MLP Paramaters
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

# Training Paramaters
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

# Graph node tensors for dnn, loss, and eval
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(
        X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(
        hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Log directory for tensorboard graphs
logdir = log_dir("mnist_dnn")

# write the file for the summary graphs
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Training with early stopping
m, n = X_train.shape

# Paramters for model
n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_deep_mnist_model"

best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        # Check and restore
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training ws interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)

    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run(
            [accuracy, loss, accuracy_summary, loss_summary],
            feed_dict={
                X: X_valid,
                y: y_valid
            })
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)
        if epoch % 5 == 0:
            print("Epoch:", epoch, "\tValidation accuracy: {:.3f}%".format(
                accuracy_val * 100), "\tLoss: {:.5f}".format(loss_val))
        saver.save(sess, checkpoint_path)
        with open(checkpoint_epoch_path, "wb") as f:
            f.write(b"%d" % (epoch + 1))
        if loss_val < best_loss:
            saver.save(sess, final_model_path)
            best_loss = loss_val
        else:
            epochs_without_progress += 5
            if epochs_without_progress > max_epochs_without_progress:
                print("Early stopping")
                break

# Display graph
tfg.board(tf.get_default_graph()).view()

# Epoch: 0 	Validation accuracy: 90.180% 	Loss: 0.36097
# Epoch: 5 	Validation accuracy: 94.940% 	Loss: 0.18178
# Epoch: 10 	Validation accuracy: 96.520% 	Loss: 0.13077
# Epoch: 15 	Validation accuracy: 97.220% 	Loss: 0.10505
# Epoch: 20 	Validation accuracy: 97.320% 	Loss: 0.09142
# Epoch: 25 	Validation accuracy: 97.580% 	Loss: 0.08275
# Epoch: 30 	Validation accuracy: 97.660% 	Loss: 0.08108
# Epoch: 35 	Validation accuracy: 97.780% 	Loss: 0.07341
# Epoch: 40 	Validation accuracy: 97.800% 	Loss: 0.07140
# Epoch: 45 	Validation accuracy: 97.980% 	Loss: 0.06950
# Epoch: 50 	Validation accuracy: 98.040% 	Loss: 0.06820
# Epoch: 55 	Validation accuracy: 97.920% 	Loss: 0.06715
# Early stopping
