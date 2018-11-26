"""
Builds two DNNS, A and B without output layer
    5 hidden layers
    100 neurons
    He initilization
    ELU activation
    Hidden layer with 10 units on both dnns
    1 neuron output layer with logistic function

"""

# Common Imports
import numpy as np

# Data Science Imports
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

# Graph Imports
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Hyperparameters

# Training operation parameters
learning_rate = 0.01
momentum = 0.95

# Inputs for layers
n_inputs = 28 * 28

# DNN He initialization hyperparameter
he_init = tf.variance_scaling_initializer()

# Model training hyperparameters
n_epochs = 100
batch_size = 500


# Declare Functions
def generate_batch(images, labels, batch_size):
    """
    Generate pairs 50% representing the same digit, or a different digit
    """
    size1 = batch_size // 2
    size2 = batch_size - size1
    # Decide same pairs or different pairs
    if size1 != size2 and np.random.rand() > 0.5:
        size1, size2 = size2, size1
    X = []
    y = []
    # Assign pairs
    while len(X) < size1:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(images), 2)
        if rnd_idx1 != rnd_idx2 and labels[rnd_idx1] == labels[rnd_idx2]:
            X.append(np.array([images[rnd_idx1], images[rnd_idx2]]))
            y.append([1])
    while len(X) < batch_size:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(images), 2)
        if labels[rnd_idx1] != labels[rnd_idx2]:
            X.append(np.array([images[rnd_idx1], images[rnd_idx2]]))
            y.append([0])
    # Randomize and distribute
    rnd_indices = np.random.permutation(batch_size)
    return np.array(X)[rnd_indices], np.array(y)[rnd_indices]


def dnn(inputs,
        n_hidden_layers=5,
        n_neurons=100,
        name=None,
        activation=tf.nn.elu,
        initializer=he_init):
    with tf.variable_scope(name, "dnn"):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(
                inputs,
                n_neurons,
                activation=activation,
                kernel_initializer=initializer,
                name="hidden%d" % (layer + 1))


# Declare classes
class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_hidden_layers=5,
                 n_neurons=100,
                 optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.01,
                 batch_size=20,
                 activation=tf.nn.elu,
                 initializer=he_init,
                 batch_norm_momentum=None,
                 dropout_rate=None,
                 random_state=None):
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None

    def _dnn(self, inputs):
        for layer in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(
                    inputs, self.dropout_rate, training=self._training)
            inputs = tf.layers.dense(
                inputs,
                self.n_neurons,
                kernel_initializer=self.initializer,
                name="hidden%d" % (layer + 1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(
                    inputs,
                    momentum=self.batch_norm_momentum,
                    training=self._training)
            inputs = self.activation(inputs, name="hidden%d_out" % (layer + 1))
        return inputs

    def _build_graph(self, n_inputs, n_outputs):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(
                False, shape=(), name="training")

        else:
            self._training = None

        dnn_outputs = self._dnn(X)

        logits = tf.layers.dense(
            dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

        xentropy = tf.nn.sparese_softmax_cross_entopy_with_logits(
            labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn_in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(
            tf.cast(correct, tf.float32), name="accuracy")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {
            gvar.op.name: value
            for gvar, value in zip(gvars, self._session.run(gvars))
        }

    def _restore_model_params(self, model_params):
        gvar_names = list(model_params.keys())
        assign_ops = {
            gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
            for gvar_name in gvar_names
        }
        init_values = {
            gvar_name: assign_op.inputs[1]
            for gvar_name, assign_op in assign_ops.items()
        }
        feed_dict = {
            init_values[gvar_name]: model_params[gvar_name]
            for gvar_name in gvar_names
        }
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
        self.close_session()

        n_inputs = X.shape[1]
        self.calsses_ = np.unique(y)
        n_outputs = len(self.classes_)

        self.calss_to_index_ = {
            label: index
            for index, label in enumerate(self.classes_)
        }
        y = np.array(
            [self.class_to_index_[label] for label in y], dtype=np.int32)

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx,
                                                  len(X) // self.batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    if self._training is not None:
                        feed_dict[self._training] = True
                    sess.run(self._training_op, feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops, feed_dict=feed_dict)
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val = sess.run(
                        [self._loss, self._accuracy],
                        feed_dict={
                            self._X: X_valid,
                            self._y: y_valid
                        })
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                    print(
                        "{}\tValidatino loss: {:.6f}\tBest loss: {:6f}\tAccuracy: {:.2f}%".
                        format(epoch, loss_val, best_loss, acc_val + 100))
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        break
                else:
                    loss_train, acc_train = sess.run(
                        [self._loss, self._accuracy],
                        feed_dict={
                            self._X: X_batch,
                            self._y: y_batch
                        })
                    print(
                        "{}\tLast training batch loss:{:.6f}\tAccuracy: {:.2f}%".
                        format(epoch, loss_train, acc_train * 100))
            if best_params:
                self._restore_model_params(best_params)
            return self

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError(
                "This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_defailt() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)

    def save(self, path):
        self._saver.save(self._session, path)


# Input placeholder
X = tf.placeholder(tf.float32, shape=(None, 2, n_inputs), name="X")
X1, X2 = tf.unstack(X, axis=1)

# Lables placeholder
y = tf.placeholder(tf.int32, shape=[None, 1])

dnn1 = dnn(X1, name="DNN_A")
dnn2 = dnn(X2, name="DNN_B")

# Concatenate outputs
dnn_outputs = tf.concat([dnn1, dnn2], axis=1)

# Extra hidden layer with 10 neurons and 1 output neuron
hidden = tf.layers.dense(
    dnn_outputs, units=10, activation=tf.nn.elu, kernel_initializer=he_init)
logits = tf.layers.dense(hidden, units=1, kernel_initializer=he_init)
y_proba = tf.nn.sigmoid(logits)

# Output probability greater or equal to 0.5, or 0
y_pred = tf.cast(tf.greater_equal(logits, 0), tf.int32)

# Cost function
y_as_float = tf.cast(y, tf.float32)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=y_as_float, logits=logits)
loss = tf.reduce_mean(xentropy)

# Create training optimizer with optimizer
optimizer = tf.train.MomentumOptimizer(
    learning_rate, momentum, use_nesterov=True)
training_op = optimizer.minimize(loss)

# Accuracy
y_pred_correct = tf.equal(y_pred, y)
accuracy = tf.reduce_mean(tf.cast(y_pred_correct, tf.float32))

# Init and saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Split the datasets 55000/5000
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

X_train1, X_train2 = X_train[5000:], X_train[:5000]
y_train1, y_train2 = y_train[5000:], y_train[:5000]

# Generate random pairs for display purposes
batch_size = 5
X_batch, y_batch = generate_batch(X_train1, y_train1, batch_size)

# Look at pairs
plt.figure(figsize=(3, 3 * batch_size))
plt.subplot(121)
plt.imshow(
    X_batch[:, 0].rehsape(28 * batch_size, 28),
    cmap="binary",
    interpolation="nearest")
plt.axis('off')
plt.subplot(122)
plt.imshow(
    X_batch[:, 1].reshape(28 * batch_size, 28),
    cmap="binary",
    interpolation="nearest")
plt.axis('off')
plt.figtext(0.0, 0.5, "{}".format((X_batch.shape, X_batch.dtype)))
plt.show()

# Generate batches
X_test1, y_test1 = generate_batch(X_test, y_test, batch_size=len(X_test))

# Do some training on image pairs
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(X_train1) // batch_size):
            X_batch, y_batch = generate_batch(X_train1, y_train1, batch_size)
            loss_val, _ = sess.run(
                [loss, training_op], feed_dict={
                    X: X_batch,
                    y: y_batch
                })
        print(epoch, "Train loss:", loss_val)
        if epoch % 5 == 0:
            acc_test = accuracy.eval(feed_dict={X: X_test1, y: y_test1})
            print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./dl_digit_comparison_model.ckpt")
