"""
Inspect namescopes
"""

# For python 2 and 3 support
from __future__ import division, unicode_literals, print_function

# Common Imports
import numpy as np
from datetime import datetime

# Data Sciecne Imports
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Graph
import tfgraphviz as tfg


# Declare Functions
def reset_graph(seed=42):
    # Seedping for output consistancy
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def fetch_batch(X, y, epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = X[indices]
    y_batch = y[indices]
    return X_batch, y_batch


# Get Data
housing = fetch_california_housing()

# Shape of Data
m, n = housing.data.shape

# Scale feature vectors
scalar = StandardScaler()
scaled_housing_data = scalar.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# Variable naming conventions handled by tensorflow
a1 = tf.Variable(0, name="a")  # a
a2 = tf.Variable(0, name="a")  # a_1

with tf.name_scope("param"):  # param
    a3 = tf.Variable(0, name="a")  # param/a

with tf.name_scope("param"):  # param_1
    a4 = tf.Variable(0, name="a")  # param_1/a

print("Name scope for graph variables")
for node in (a1, a2, a3, a4):
    print("name variable or name_scope =", node.op.name)


tfg.board(tf.get_default_graph()).view()

reset_graph()

# Create logging with timestamps
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Perform minibatch gradient descent name_scopes
n_epochs = 1000
learning_rate = 0.01

# Declare tensor nodes
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(
    tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

# Define a loss name_scope
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

# Instantiate an optimizer and training operations
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# Define an initialization variable
init = tf.global_variables_initializer()

# String Scaler Tensor Nodes for files and graphs
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Perform Batch Gradient Descent in a tensorflow session
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(scaled_housing_data_plus_bias,
                                           housing.target.reshape(-1, 1),
                                           epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={
                    X: X_batch,
                    y: y_batch
                })
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.flush()
file_writer.close()

tfg.board(tf.get_default_graph()).view()

print("Best theta:")
print(best_theta)

print("node name_scope", error.op.name)
print("node name_scope", mse.op.name)
