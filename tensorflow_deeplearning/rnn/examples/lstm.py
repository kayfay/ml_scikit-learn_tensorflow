# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np
import os
import webbrowser

# Data Science Imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Graph Imports
import tfgraphviz as tfg
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Config
PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Hyperparameters
# Neural Network Hyperparameters
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
n_layers = 3

# Optimization Hyperparameters
learning_rate = 0.001

# Batch parameters
n_epochs = 10
batch_size = 150


# Declare Functions
def save_fig(fig_id, tight_layout=True):
    if not os.path.exists('images'):
        os.makedirs('images')
    path = os.path.join(PROJECT_ROOT_DIR, 'images', fig_id + ".png")
    plt.savefig(path, format='png', dpi=300)


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = bytes("<stripped %d bytes>" % size,
                                              'utf-8')
    return strip_def


def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(
                s[1:])
    return res_def


def show_graph(graph_def=tf.get_default_graph().as_graph_def(),
               graph_title='graph.html',
               max_const_size=32):
    """Visualize TensorFlow graph."""

    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)

    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html"
        onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"><tf-graph-basic>
        </div>
    """.format(
        data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1800px;height:1620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))

    os.makedirs('html', exist_ok=True)

    graph_file = os.path.join(PROJECT_ROOT_DIR, 'html', graph_title)

    with open(graph_file, 'w') as g:
        g.write(iframe)

    webbrowser.open_new_tab(graph_file)

    # Use Graphviz to view graph
    graph = tf.get_default_graph()
    tfg.board(graph, depth=2).view()


# Neural Network Nodes
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

lstm_cells = [
    tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
    for layer in range(n_layers)
]
multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initalize nodes of global scope
init = tf.global_variables_initializer()

# Get Data
mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

# Train Network
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print("Epoch", epoch, "Train accuracy = ", acc_train,
              "Test accuracy =", acc_test)

# Examples of alternative cells
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, use_peepholes=True)
gru_cell = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)

# Display layers
print("Layers \n", states, "\n", top_layer_h_state)

# Generate neural network graph
show_graph(graph_title="lstm_graph")
