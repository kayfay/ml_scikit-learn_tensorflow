"""
Batch Gradient Descent
"""
# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np

# Data Science Imports
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

# Declare functions


def reset_graph(seed=42):
    # Seeding for output consistancy
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def partial_derivatives(a, b):
    z = 0
    for i in range(100):
        z = a * np.cos(z + i) + z * np.sin(b - i)
    return z


# Set seed
reset_graph()

# TensorFlow Opeartions (ops) inputs/outputs
# Constants and variables as source ops
# Tensors a type and shape

# Get data
housing = fetch_california_housing()

# Shape of data
m, n = housing.data.shape

# Scale feature vectors
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# Shape and means of data
print(scaled_housing_data_plus_bias.mean(axis=0))
print(scaled_housing_data_plus_bias.mean(axis=1))
print(scaled_housing_data_plus_bias.mean())
print(scaled_housing_data_plus_bias.shape)

# Manually computing the gradients
n_epochs = 1000
learning_rate = 0.01

# Create tensorflow constants nodes for X and y matrices
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# Initalize a random theta node
theta = tf.Variable(
    tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
# Create a node for the predictions
y_pred = tf.matmul(X, theta, name="predictions")
# Compute residual errors
error = y_pred - y
# Mean squared error node
mse = tf.reduce_mean(tf.square(error), name="mse")
# Gradient computation
gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# Perform training operations
training_op = tf.assign(theta, theta - learning_rate * gradients)

# Initalization variable
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):  # Iterate over 1000 epochs
        if epoch % 100 == 0:  # Display for 10 cycles
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)  # Perform assignment of training

    best_theta = theta.eval()  # Evaluate the theta operation

# Using autodiff (the same as above)
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(
    tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# Using tensorflow builtin autodiff gradient computation
gradients = tf.gradients(mse, [theta])[0]

training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

# Partial Derivatives
reset_graph()

a = tf.Variable(0.2, name="a")
b = tf.Variable(0.3, name="b")
z = tf.constant(0.0, name="z0")

for i in range(100):
    z = a * tf.cos(z + i) + z * tf.sin(b - i)

grads = tf.gradients(z, [a, b])
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print(z.eval())
    print(sess.run(grads))

# Using function
reset_graph()
dz = partial_derivatives(0.2, 0.3)

# Using a gradient descent optimizer
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(
    tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# Initalize optimizer with 0.01 l
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Minimize based on mse node
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

# Using a momentum optimizer
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(
    tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# Node for initalizing the optimizer
optimizer = tf.train.MomentumOptimizer(
    learning_rate=learning_rate, momentum=0.9)

# Training optimizer
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        sess.run(training_op)

    best_theta = theta.eval()

