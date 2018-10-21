"""
Resuing models from other frameworks
Find the initializer assignment opeartion
get its second input, the initialization value
run the initalizer with replaced values

"""

import tensorflow as tf
import numpy as np


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


n_inputs = 2
n_hidden1 = 3

original_w = [[1., 2., 3.], [4., 5.,
                             6.]]  # load weights from the other framework
original_b = [7., 8., 9.]  # load the biases from the other framework

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
# ... build the reset of the model ...

# Get the assignment nodes for the hidden1 variables
graph = tf.get_default_graph()
assign_kernel = graph.get_operation_by_name("hidden1/kernel/Assign")
assign_bias = graph.get_operation_by_name("hidden1/bias/Assign")
init_kernel = assign_kernel.inputs[1]
init_bias = assign_bias.inputs[1]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init, feed_dict={init_kernel: original_w, init_bias: original_b})
    # ... train the model on the new task
    print(hidden1.eval(feed_dict={X: [[10.0, 11.0]]}))

# A more verbose and less efficient approach to set explicitly

reset_graph()

n_inputs = 2
n_hidden1 = 3

original_w = [[1., 2., 3.], [4., 5., 6.]]
original_b = [7., 8., 9.]

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
# ... Build out the reset of the model

# Get the reset of the variables of layer hidden
with tf.variable_scope("", default_name="", reuse=True):
    hidden1_weights = tf.get_variable("hidden/kernel")
    hidden1_biases = tf.get_variable("hidden1/bias")

# Create dedicated placeholders and assignment nodes
original_weights = tf.placeholder(tf.float32, shape=(n_inputs, n_hidden1))
original_biases = tf.placeholder(tf.float32, shape=n_hidden1)
assign_hidden1_weights = tf.assign(hidden1_weights, original_weights)
assign_hidden1_biases = tf.assign(hidden1_biases, original_biases)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(assign_hidden1_weights, feed_dict={original_weights: original_w})
    sess.run(assign_hidden1_biases, feed_dict={original_biases: original_b})
    # ... train the model out on the task
    print(hidden1.eval(feed_dict={X: [[10.0, 11.0]]}))

#  Using get_collectio() to select variables
tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden1")

#  Using get_tensor_by_name()
tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")

tf.get_default_graph().get_tensor_by_name("hidden1/bias:0")
