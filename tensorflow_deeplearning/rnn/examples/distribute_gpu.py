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
n_inputs = 5
n_steps = 20
n_neurons = 100

# Declare Functions


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


# Classes


class DeviceCellWrapper(tf.nn.rnn_cell.RNNCell):
    """
    Provides functionality to distribute cells across multiple GPUs
    Similar to tf.nn.rnn_cell.DeviceWrapper
    """

    def __init__(self, device, cell):
        self._cell = cell
        self._device = device

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with tf.device(self._device):
            return self._cell(inputs, state, scope)


# Neural Network
X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs])
devices = ["/cpu:0", "/cpu:0", "/cpu:0"]  # gpu:0, gpu:1, gpu:2
cells = [
    DeviceCellWrapper(dev, tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons))
    for dev in devices
]
multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# Initialize Variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print(
        sess.run(outputs, feed_dict={X: np.random.rand(2, n_steps, n_inputs)}))
