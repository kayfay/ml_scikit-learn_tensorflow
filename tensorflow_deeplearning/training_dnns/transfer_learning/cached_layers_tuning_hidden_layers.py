"""
Using MNIST dataset

A DNN Using a previous model of pretrained hidden layers
    Adjust layers for accuracy tuning

Freeze and unfreeze layers to tune model
    Compare training times and accuracy
    Cache frozen layers for time
    Reduce Hidden layers

Replaces softmax outplayer with a new one

Unfreezes the top two hidden layers for model tuning

"""

# Common Imports
import numpy as np
import time

# Data Science Imports
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

# Graph Imports

# Config

# Hyperparameters

# Optimization Hyperparameters
learning_rate = 0.01

# Softmax layer output
n_outputs = 5

# Time a training session, uses early stopping, restores best model, reinit output layer
# Specific Session Hyperparameters
# Training Hyperparameters
n_epochs = 1000
batch_size = 20

# Earlystopping Hyperparameters
max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

# For dnn function use He initialization
he_init = tf.variance_scaling_initializer()

# Declare Functions


def sample_n_instance_per_class(X, y, n=100):
    Xs, ys = [], []
    for label in np.unique(y):
        idx = (y == label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)


# Declare Classes
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
        """ Initialize the DNNClassifier with hyperparameters """
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.random_state = random_state
        self._session = None

    def _dnn(self, inputs):
        """ Build hidden layers with support for
            batch normaliaztion and dropout """
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
        """ Build model scores """
        if self.random_state is not None:
            tf.set_random_state(self.random_seed)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(
                False, shape=(None), name="training")
        else:
            self._training = None

        dnn_outputs = self._dnn(X)

        logits = tf.layers.dense(
            dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y)

        loss = tf.reduce_mean(xentropy, name="loss")

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(
            tf.cast(correct, tf.float32), name="accuracy")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Create instance variables for operations
        self._X, self._y, = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        """ Get all variable values
        (for early stopping, instead of saving to disk) """
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {
            gvar.op.name: value
            for gvar, value in zip(gvars, self._session.run(gvars))
        }

    def _restore_model_params(self, model_params):
        """ Set variables to given values """
        gvar_names = list(model_params.keys())
        assign_ops = {
            gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
            for gvar_name in gvar_names
        }

        init_values = {{
            gvar_name: assign_op.inputs[1]
            for gvar_name, assign_op in assign_ops.items()
        }}

        feed_dict = {
            init_values[gvar_name]: model_params[gvar_name]
            for gvar_name in gvar_names
        }

        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
        """ Fit the model scores to the training set.
        Use early stopping if X_valid and y_valid are provided """
        self.close_session()

        # Infer n_inputs and n_outputs from the training set
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)

        # Translates the labels vector to a vector of sorted class
        # indices, containing integers from 0 to n_outputs - 1
        # For example, if y is equal to [8, 8, 9, 5, 7, 6, 6, 6],
        # then the sorted class labels (self.classes_) will be
        # equal to [5, 6, 7, 8, 9], and the labels vector
        # will be translated to [3, 3, 4, 0, 2, 1, 1, 1]

        self.class_to_index_ = {
            label: index
            for index, label in enumerate(self.classes_)
        }

        y = np.array(
            [self.class_to_index_[label] for label in y], dtype=np.int32)

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            # extra ops for batch normalization
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Earlystopping hyperparameters
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        # Model Training
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
                        "{\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".
                        format(epoch, loss_val, best_loss, acc_val * 100))
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping")
                        break

                else:
                    loss_train, acc_train = sess.run(
                        [self._loss, self.accuracy],
                        feed_dict={
                            self._X: X_batch,
                            self._y: y_batch
                        })

            if best_params:
                self._restore_model_params(best_params)
            return self

        def predict_proba(self, X):
            if not self._session:
                raise NotFittedError("This %s instance is not fitted yet" %
                                     self.__class__.__name__)
            return self._Y_proba.eval(feed_dict={self._X: X})

        def predict(self, X):
            class_indices = np.argmax(self.predict_proba(X), axis=1)
            return np.array([[self.classes_[class_index]]
                             for class_index in class_indices], np.int32)

        def save(self, path):
            self._saver.save(self._session, path)


# Dataset config
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# Load the models graph
restore_saver = tf.train.import_meta_graph("./dl_best_mnist_model_0_to_4.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

# Load a softmax output ontop of the 4th hidden layer
hidden4_out = tf.get_default_graph().get_tensor_by_name("hidden4_out:0")

# Add new layer implementing softmax
logits = tf.layers.dense(
    hidden4_out, n_outputs, kernel_initializer=he_init, name="new_logits")
Y_proba = tf.nn.softmax(logits)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# Freeze layers except for new output layer
output_layer_vars = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope="new_logits")

# Unfreeze the top two hidden layers and continue training
unfrozen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[34]|new_logits")
optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam3")
training_op = optimizer.minimize(loss, var_list=unfrozen_vars)

# Correct probabiilities, accuracy metric
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# Initialize variables and a save object
init = tf.global_variables_initializer()
four_frozen_saver = tf.train.Saver()
two_frozen_saver = tf.train.Saver()

# Train using 100 images per digits
X_train2_full = X_train[y_train >= 5]
y_train2_full = y_train[y_train >= 5] - 5
X_valid2_full = X_valid[y_valid >= 5]
y_valid2_full = y_valid[y_valid >= 5] - 5
X_test2 = X_test[y_test >= 5]
y_test2 = y_test[y_test >= 5] - 5

# 100 / 30 split instances per class in the training / test set
X_train2, y_train2 = sample_n_instance_per_class(
    X_train2_full, y_train2_full, n=100)
X_valid2, y_valid2 = sample_n_instance_per_class(
    X_valid2_full, y_valid2_full, n=30)

# Run the optimization and process
with tf.Session() as sess:
    init.run()
    four_frozen_saver.restore(sess, "./dl_mnist_model_5_to_9_four_frozen")

    t0 = time.time()

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indices in np.array_split(rnd_idx,
                                          len(X_train2) // batch_size):
            # Add cached layers
            X_batch, y_batch = X_train2[rnd_indices], y_train2[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run(
            [loss, accuracy], feed_dict={
                X: X_valid2,
                y: y_valid2
            })
        if loss_val < best_loss:
            save_patch = four_frozen_saver.save(
                sess, "./dl_mnist_model_5_to_9_two_frozen")
            best_loss = loss_val
            checks_without_progress = 0

        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break

        print(
            "{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".
            format(epoch, loss_val, best_loss, acc_val * 100))

    t1 = time.time()
    print("Total training time: {:.1f}s".format(t1 - t0))

with tf.Session() as sess:
    four_frozen_saver.restore(sess, "./dl_mnist_model_5_to_9_two_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
