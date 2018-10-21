"""
Feeding data into a training algorithm, i.e., mini-batch gradient descent
"""

# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np

# Data Science Imports
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Declare functions


def reset_graph(seed=42):
    # Seeding for output consistancy
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch


# Set seed
reset_graph()

# Get data
housing = fetch_california_housing()

# Shape of data
m, n = housing.data.shape

# Scale feature vectors
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# Placeholder nodes
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
# Perform evaluation in session
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print(B_val_1)
print(B_val_2)

# Mini-batch Gradient
# Use placeholder nodes to incrementally feed in batches
n_epochs = 1000
learning_rate = 0.01

reset_graph()
# Create placeholder nodes for feeding in data
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

# A node for init of randomize thetas
theta = tf.Variable(
    tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
# Node for predictions
y_pred = tf.matmul(X, theta, name="predictions")
# Computations for MSE
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# Optimization algorithm
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Minimize operator for MSE
training_op = optimizer.minimize(mse)

# Initialize global variable object
init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):  # Perform batch operations
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

print("Theta values\n", best_theta)
