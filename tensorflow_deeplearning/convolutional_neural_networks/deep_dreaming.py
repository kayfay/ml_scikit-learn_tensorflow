"""
github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
Generative convolutional neural network image generation
"""
# Python 2 and 3 imports
from __future__ import print_function

# Common Imports
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import zipfile
import sys
from six.moves import urllib
import tfgraphviz as tfg
import webbrowser

# Data science imports
import tensorflow as tf

# Config
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
TF_MODELS_URL = "https://storage.googleapis.com/download.tensorflow.org/models"
INCEPTION_URL = TF_MODELS_URL + "/inception5h.zip"
INCEPTION_PATH = os.path.join(PROJECT_DIR, "datasets", "inception")


# Functions
def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()


def fetch_inception(url=INCEPTION_URL, path=INCEPTION_PATH):
    os.makedirs(path, exist_ok=True)
    zip_path = os.path.join(path, "inception5h.zip")
    if os.path.exists(os.path.join(path, "inception5h.zip")):
        urllib.request.urlretrieve(url, zip_path, reporthook=download_progress)
        inception_zip = zipfile.ZipFile(zip_path, 'r')
        inception_zip.extractall(path=path)
        inception_zip.close()


# TF Graph Visualization functions
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


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""

    graph_file = open('graph.html', 'w')

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
    display(HTML(iframe))

    with graph_file as g:
        g.write(iframe)

    filename = os.path.join(PROJECT_DIR, 'graph.html')
    webbrowser.open_new_tab(filename)

    # Use Graphviz to view graph
    graph = tf.get_default_graph()
    tfg.board(graph, depth=2).view()


# Naive feature visualization functions


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def visstd(a, s=0.1):
    """Normalize the image range for visualization"""
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5


def T(layer):
    """Helper function for getting layer output tensor"""
    return graph.get_tensor_by_name("import/%s:0" % layer)


def render_naive(t_obj, img0, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj)  # define the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0]  # auto diff magic

    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input: img})
        # Normalize the gradient for equal steps by layer
        g /= g.std() + 1e-8
        img += g * step
        print(score, end=' ')
    clear_output()
    showarray(visstd(img))


# Model
print(PROJECT_DIR)
fetch_inception()
model_fn = os.path.join(INCEPTION_PATH, "tensorflow_inception_graph.pb")

# Create TensorFlow session and load model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

# Explore patterns learned
# Generate images that maximize sums of activations
# Of a channel of a convolutional layer from the network
# Each layer outputs 10s to 100s of feature channels

layers = [
    op.name for op in graph.get_operations()
    if op.type == 'Conv2D' and 'import/' in op.name
]
feature_nums = [
    int(graph.get_tensor_by_name(name + ':0').get_shape()[-1])
    for name in layers
]

print("Number of layers", len(layers))
print("Total number of feature channels:", sum(feature_nums))

# Visualizing the network graph.
# Expand 'mixed' nodes
tmp_def = rename_nodes(graph_def, lambda s: "/".join(s.split('_', 1)))
if os.path.exists(os.path.join(PROJECT_DIR, 'graph.html')):
    show_graph(tmp_def)

# Naive feature visualization
# Image-space gradient ascent
# Picking some internal layer, using outputs before applying ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139  # arbitrary feature channel

# Create gray image with noise
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

render_naive(T(layer)[:, :, :, channel], img_noise)
