# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np
import os
import webbrowser

# Data Science Imports
import tensorflow as tf

# Graph Imports
import tfgraphviz as tfg
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Config
PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Hyperparameters

# Network Hyperparameters
n_steps = 20  # unrolled over 20 timesteps
n_inputs = 1
n_neurons = 100
n_outputs = 1

# Optimization Hyperparameters
learning_rate = 0.001

# Batch Training Parameters
n_iterations = 1500
batch_size = 50

# Timeseries points
t_min, t_max = 0, 30
resolution = 0.1


# Declare Functions
def save_fig(fig_id, tight_layout=True):
    os.makedirs('images', exist_ok=True)
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


def show_graph(graph_def, graph_title='graph.html', max_const_size=32):
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


def time_series(t):
    """
    Calculates a timeseries

    Uses time point, t of sine waves at angle t
    """
    return t * np.sin(t) / 3 + 2 * np.sin(t * 5)


def next_batch(batch_size, n_steps):
    segment_num_samples = t_max - t_min - n_steps * resolution
    t0 = np.random.rand(batch_size, 1) * segment_num_samples
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(
        -1, n_steps, 1)


# Neural Network
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

# Create basic cell
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(
    num_units=n_neurons, activation=tf.nn.relu)

# 'state' is tensor of shape [batch_size, cell_state_size]
rnn_outputs, states = tf.nn.dynamic_rnn(rnn_cell, X, dtype=tf.float32)

# rnn layers
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

# Optimization
loss = tf.reduce_mean(tf.square(outputs - y))  # mean squared error
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

# Initialize graph nodes
init = tf.global_variables_initializer()

# Neural Network Graph Saver
saver = tf.train.Saver()

# Generate some data
stop_value = 12.2 + resolution * (n_steps + 1)
num_samples = n_steps + 1
t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))
t_instance = np.linspace(12.2, stop_value, num_samples)

# Training
with tf.Session() as sess:
    os.makedirs('saved_models', exist_ok=True)
    saver.restore(sess, "./saved_models/my_time_series_model")

    sequence = [0.] * n_steps
    for iteration in range(300):
        X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence.append(y_pred[0, -1, 0])

plt.figure(figsize=(8, 4))
plt.plot(np.arange(len(sequence)), sequence, "b-")
plt.plot(t[:n_steps], sequence[:n_steps], "b-", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")
save_fig('generation_sequence_plot')
plt.show()
