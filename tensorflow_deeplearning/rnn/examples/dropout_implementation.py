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
# Neural Network Hyperparameters
n_inputs = 1
n_neurons = 100
n_layers = 3
n_steps = 20
n_outputs = 1

# Optimization Hyperparameters
learning_rate = 0.01

# Batch Training Hyperparameters
n_iterations = 1500
batch_size = 50

# Dropout Hyperparameters
train_keep_prob = 0.5

# Timeseries points
t_min, t_max = 0, 30
resolution = 0.1

# Declare Functions


def save_fig(fig_id, tight_layout=True):
    if not os.path.exists('images'):
        os.makedirs('images')
    path = os.path.join(PROJECT_ROOT_DIR, 'images', fig_id + ".png")
    plt.savefig(path, format='png', dpi=300)


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
    return t * np.sin(t) / 3 + 2 * np.sin(t * 5)


def next_batch(batch_size, n_steps):
    segment_num_samples = t_max - t_min - n_steps * resolution
    t0 = np.random.rand(batch_size, 1) * segment_num_samples
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(
        -1, n_steps, 1)


# Generate data
stop_value = 12.2 + resolution * (n_steps + 1)
num_samples = n_steps + 1
t_instance = np.linspace(12.2, stop_value, num_samples)

# Neural network
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, shape=[None, n_steps, n_outputs])

# Dropout
# Add dropout, add between layers with DropoutWrapper
keep_prob = tf.placeholder_with_default(1.0, shape=())
cells = [
    tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
    for layer in range(n_layers)
]
cells_drop = [
    tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    for cell in cells
]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(cells_drop)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# more stuff
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Train model
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        _, mse = sess.run(
            [training_op, loss],
            feed_dict={
                X: X_batch,
                y: y_batch,
                keep_prob: train_keep_prob
            })
        if iteration % 100 == 0:
            print(iteration, "Training MSE:", mse)

    os.makedirs('saved_models', exist_ok=True)
    saver.save(sess, "./saved_models/my_dropout_time_series_model")

# Restore session for whatever reason
with tf.Session() as sess:
    saver.restore(sess, "./saved_models/my_dropout_time_series_model")

    X_new = time_series(
        np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})

plt.title("Testing the model", fontsize=14)
plt.plot(
    t_instance[:-1],
    time_series(t_instance[:-1]),
    "bo",
    markersize=10,
    label="instance")
plt.plot(
    t_instance[1:],
    time_series(t_instance[1:]),
    "w*",
    markersize=10,
    label="target")
plt.plot(
    t_instance[1:], y_pred[0, :, 0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")
plt.show()
