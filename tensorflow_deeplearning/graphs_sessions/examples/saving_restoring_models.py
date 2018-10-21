"""
Saving and restoring models
"""

# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Data Science Imports
import tensorflow as tf

# Declare functions


def reset_graph(seed=42):
    # Seeding for output consistancy
    tf.reset_default_graph()


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

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(
    tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)

    best_theta = theta.eval()

save_path = saver.save(sess, '/tmp/my_model_final.ckpt')

print("\n", best_theta)

with tf.Session) as ses:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval)o

print("Compare both", np.allclose(best_theta, best_theta_restored))

saver = tf.train.Saver({"weights":theta})
