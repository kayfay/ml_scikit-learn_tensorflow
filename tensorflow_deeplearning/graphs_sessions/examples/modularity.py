"""
Modularity examples
"""
# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Data Science Imports
import tensorflow as tf

# Graph
import tfgraphviz as tfg


# Declare Functions
def relu(X):
    # Rectified Linear Unit
    # h_{w,b} (X) = max ( X \cdot w + b, 0 )
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, 0, name="relu")


# Create nodes
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]      # Compute multiple ReLU
output = tf.add_n(relus, name="output")  # Sum a list of tensors

# Write out files
file_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())
file_writer.close()

# Generate Graph
tfg.board(tf.get_default_graph()).view()
