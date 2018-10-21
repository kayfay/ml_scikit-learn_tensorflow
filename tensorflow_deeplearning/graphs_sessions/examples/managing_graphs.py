"""
Managing graphs construction phase
"""

# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np

# Data Science Imports
import tensorflow as tf


# Declare functions


def reset_graph(seed=42):
    # Seedping for output consistancy
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# Set seed
reset_graph()

# Create node on the default graph
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()  # True is on default graph

# Create a new graph
graph = tf.Graph()  # For managing multiple default graphs

with graph.as_default():  # For assigning to a specified graph
    x2 = tf.Variable(2)

x2.graph is graph  # True is a graph
x2.graph is tf.get_default_graph()  # False is the global default

tf.reset_default_graph()  # To reset the default graph

# Evaluation occurs twice
# To evaluate w, and x dependant on w
# Then again to run the graph evaluation
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15

# Node values are dropped between graph runs, except variables
# Sessions que, maintain state, open and close node values
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15

