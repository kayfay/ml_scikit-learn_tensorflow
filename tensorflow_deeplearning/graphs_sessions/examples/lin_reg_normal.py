"""
Linear Regression model with TensorFlow
"""
# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np

# Data Science Imports
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

# Declare functions


def reset_graph(seed=42):
    # Seeding for output consistancy
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# Set seed
reset_graph()

# TensorFlow Opeartions (ops) inputs/outputs
# Constants and variables as source ops
# Tensors a type and shape

# Get data
housing = fetch_california_housing()

# Shape of data
m, n = housing.data.shape

# Add a bias column
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# Create constant variables
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
# Linear Regression with the normal equation
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

# Evaluate expression
with tf.Session() as sess:
    theta_value = theta.eval()

# Compared to NumPy
X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
# Linear Regression with the normal equation
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Compared with Scikit-Learn
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))
theta_sklearn = np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T]

print('tf:\n', theta_value)
print('np:\n', theta_numpy)
print('sk:\n', theta_sklearn)
