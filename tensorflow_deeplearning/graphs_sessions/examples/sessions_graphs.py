"""
Create a graph construction phase
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

# Create a computation graph
x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')

f = x * x * y + y + 2
print("\nA tensor object containing computation graph:\n", f)

# Create a session
sess = tf.Session()
# Initalize the variables
sess.run(x.initializer)
sess.run(y.initializer)
# Evaluate and close the session
result = sess.run(f)
sess.close()
print(" f = x * x * y + y + 2 = ", result)

# Instead of closing sess.run()
with tf.Session() as sess:
    x.initializer.run()  # tf.get_default_session().run(x.initializer)
    y.initializer.run()
    result = f.eval()  # tf.get_default_session().run(f)

# Or initalize all of the variables at once
# Create node in graph for runtime
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()  # Initalize during session
    result = f.eval()

# Using an interactive session, for jupyter, ipython, or an interactive shell

init = tf.global_variables_initializer()  # prepare an initialization node

sess = tf.InteractiveSession()
init.run()  # initalize all of the variables
result = f.eval()
sess.close()
