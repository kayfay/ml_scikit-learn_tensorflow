"""
Logistic Regression with Mini-Batch Gradient Descent using Tensorflow
"""


# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Graph Imports
import matplotlib.pyplot as plt
# Common Imports
import numpy as np
# Data science Imports
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.metrics import precision_score, recall_score

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Config


# Declare Functions
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def random_batch(X_train, y_train, batch_size):
    """Pick random instances from 1st and 2nd arguments"""
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


# Create a dataset
m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

# Plot the dataset
plt.plot(
    X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
plt.plot(
    X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")

plt.legend()
plt.show()

# Add an extra bias feature to every instance
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]

# Reshape y_train into a column vector
y_moons_column_vector = y_moons.reshape(-1, 1)

# Split training and test set
test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]

# Reset default graph
reset_graph()

# Input paramaters
n_inputs = 2  # moons dataset has 2 input features
epsilon = 1e-7  # for a small episolon to prevent errors
learning_rate = 0.01

# Create Node tensors
# Create X and y placeholders
X = tf.placeholder(
    tf.float32, shape=(None, n_inputs + 1), name="X")  # No instances, 3 rows
y = tf.placeholder(
    tf.float32, shape=(None, 1), name="y")  # No instances, 1 row
theta = tf.Variable(
    tf.random_uniform(  # random uniform -1 by 1 disribution of 0
        [n_inputs + 1, 1],
        -1.0,
        1.0,
        seed=42),
    name="theta")  # for 3 by 1 matrix initialization

# Compute values
logits = tf.matmul(X, theta, name="logits")  # product of weights and values

# Sigmoid function
# y_proba = 1 / (1 + tf.exp(-logits))                      # Function by hand
y_proba = tf.sigmoid(logits)  # Builtin function

# Loss function
# loss = -tf.reduce_mean(              # Loss function by hand
#    y * tf.log(y_proba + epsilon) + 1 - y) * tf.log(1 - y_proba + epsilon))
loss = tf.losses.log_loss(y, y_proba)  # tf log loss function

# Optimization function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# Variable initialization
init = tf.global_variables_initializer()

# Train model
# Model paramaters
n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval({X: X_test, y: y_test})
        if epoch % 100 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)

    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})

# Prediction score estimates > 0.5 threshold
y_pred = (y_proba_val >= 0.5)

# Precision/Recall metrics
print("Precision Score:", precision_score(y_test, y_pred))
print("Recall Score", recall_score(y_test, y_pred))

# Plot predictions
y_pred_idx = y_pred.reshape(-1)  # a 1D array rather than a column vector
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label="Poisitive")
plt.plot(
    X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label="Negative")
plt.legend()
plt.show()
