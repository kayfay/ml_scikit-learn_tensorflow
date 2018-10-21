"""
Using MNIST dataset

DNN with 5 Hidden Layers 100 Neurons each
    Uses He Initialization
    Uses an ELU activation function

    Uses a DNN Function, logit and softmax for probabilities

Adds Adam optimization and early stopping
    Trained on digits 0-4
    Uses a softmax output layer with 5 neurons
    Saves interval checkpoints and final model_scores


Adds a DNNClassifier class
    Uses cross-validation to tune hyperparameters
    Uses precision for a metric

Adds Batch Normalization
    Compares learning curves for TensorBoard
        Faster convergence
        Better model_scores

Checks if the model_scores is overfitting
Uses dropout on each layer
"""

# Common Imports
import numpy as np

# Data Science Imports
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Graph Imports

# Config

# Hyperparameters

# Network hyperparameters
n_inputs = 28 * 28
n_outputs = 5

# Optimizer Hyperparameters
learning_rate = 0.01

# Model training session hyperparameters
n_epochs = 1000
batch_size = 20

# Models Accuracy
model_scores = list()

# Early stopping hyperameters
max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

# For dnn function use He initiation
he_init = tf.variance_scaling_initializer()

# Declare Functions


def dnn(inputs,
        n_hidden_layers=5,
        n_neurons=100,
        name=None,
        activation=tf.nn.elu,
        initializer=he_init):
    """ Network layer function to automate the dnn creation process

       param: inputs e.g. X
       param: n_hidden_layers e.g. 5 number of layers in network
       param: n_neurons e.g. 100 neurons in each layer
       param: name e.g. hidden1 name of layer
       param: activation function eg relu, leaky ReLU, tanh, logistic, softmax
       param: initializer weight variable variance and randomization method
    """

    with tf.variable_scope(name, "dnn"):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(
                inputs,
                n_neurons,
                activation=activation,
                kernel_initializer=initializer,
                name="hidden%d" % (layer + 1))

            return inputs


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def leaky_relu(alpha=0.01):
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)

    return parametrized_leaky_relu


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
        """Initialize the DNNClassifier by storing all the hyperparameters"""
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
        """ Build the hidden layers with support for
            batch normalization and dropout"""
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
        """Build the model_scores"""
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(
                False, shape=(), name='training')
        else:
            self._training = None

        dnn_outputs = self._dnn(X)

        logits = tf.layers.dense(
            dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)

        loss = tf.reduce_mean(xentropy, name="loss")

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(
            tf.cast(correct, tf.float32), name="accuracy")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available
        # easily through instance variables
        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        """ Get all variable values (used for early stopping,
            faster than saving to disk) """
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {
            gvar.op.name: value
            for gvar, value in zip(gvars, self._session.run(gvars))
        }

    def _restore_model_params(self, model_params):
        """Set all variables to the given values
        (for early stopping, faster than loading from disk)"""
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
        """ Fit the model_scores to the training set.
            If X_valid and y_valid are provided, use early stopping """
        self.close_session()

        # infer n_inputs and n_outputs from the training set
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

        # Earlystopping hyperparametrs
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
                        "{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".
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
                    print(
                        "{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".
                        format(epoch, loss_train, acc_train * 100))
            if best_params:
                self._restore_model_params(best_params)
            return self

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError(
                "This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)

    def save(self, path):
        self._saver.save(self._session, path)


# Dataset Config
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# Build graph tensor nodes

# THIS IS THE PART WHERE EACH NODE PROCESS GETS INITALIZED
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

dnn_outputs = dnn(X)

logits = tf.layers.dense(
    dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
Y_proba = tf.nn.softmax(logits, name="Y_proba")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")

# Adam optimization and early stopping implementation

# Graph the nodes training opeartion and cost function
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")

# Create accuracy from correct classifications
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# Instantiate the variables and the save function
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Create the training set, validation, and the test set where digits 0-4
X_train1 = X_train[y_train < 5]
y_train1 = y_train[y_train < 5]
X_valid1 = X_valid[y_valid < 5]
y_valid1 = y_valid[y_valid < 5]
X_test1 = X_test[y_test < 5]
y_test1 = y_test[y_test < 5]

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train1))
        for rnd_indices in np.array_split(rnd_idx,
                                          len(X_train1) // batch_size):
            X_batch, y_batch = X_train1[rnd_indices], y_train1[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run(
            [loss, accuracy], feed_dict={
                X: X_valid1,
                y: y_valid1
            })
        if loss_val < best_loss:
            save_path = saver.save(sess, "./dl_mnist_model_0_to_4.ckpt")
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

# Test accuracy
with tf.Session() as sess:
    saver.restore(sess, "./dl_mnist_model_0_to_4.ckpt")
    acc_test = accuracy.eval(feed_dict={X: X_test1, y: y_test1})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

# Test accuracy without dropout or batch norm
dnn_clf = DNNClassifier(random_state=42)
dnn_clf.fit(
    X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)

# Compare accuracy from both tests
y_pred = dnn_clf.predict(X_test1)
acc_wo = str("Accuracy from w/o dropout or batch normaliaztion: {:.2f}".format(
    accuracy_score(y_test1, y_pred)))
model_scores.append(acc_wo)

# Implement randomized search with cross validation
param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation":
    [tf.nn.relu, tf.nn.elu,
     leaky_relu(alpha=0.01),
     leaky_relu(alpha=0.1)],
}

# Perform randomized search with cross validation for accuracy best params
rnd_search = RandomizedSearchCV(
    DNNClassifier(random_state=42),
    param_distribs,
    n_iter=50,
    random_state=42,
    verbose=2)
rnd_search.fit(
    X_train1, y_train1, X_valid=X_valid1, y_valid=y_valid1, n_epochs=1000)

# Displaying the best paramaters and accuracy
rnd_search.best_params_
y_pred = rnd_search.predict(X_test1)
acc_rnd_search = str(
    "Accuracy from best paramaters in randomized search:".format(
        accuracy_score(y_test1, y_pred)))
model_scores.append(acc_rnd_search)

# Save model_scores
rnd_search.best_estimator_.save("./dl_best_mnist_model_0_to_4")

# Adding Batch Normalization and compare learning curves
dnn_clf = DNNClassifier(
    activation=leaky_relu(alpha=0.1),
    batch_size=500,
    learning_rate=0.01,
    n_neurons=140,
    random_state=42)
dnn_clf.fit(
    X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)

# Accuracy on the test set
y_pred = dnn_clf.predict(X_test1)
acc_dnn = str("Best fitted parameters on test set:".format(
    accuracy_score(y_test1, y_pred)))
model_scores.append(acc_dnn)

# Batch normalization
dnn_clf_bn = DNNClassifier(
    activation=leaky_relu(alpha=0.1),
    batch_size=500,
    learning_rate=0.01,
    n_neurons=90,
    random_state=42,
    batch_norm_momentum=0.95)
dnn_clf_bn.fit(
    X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)

# Accuracy on batch normalization
y_pred = dnn_clf_bn.predict(X_test1)
acc_bn = str("Same model_scores with batch normalization added:".format(
    accuracy_score(y_test1, y_pred)))
model_scores.append(acc_bn)

# Batch normalization randomized search with cross val
param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation":
    [tf.nn.relu, tf.nn.elu,
     leaky_relu(alpha=0.01),
     leaky_relu(alpha=0.1)],
    "batch_norm_momentum": [0.9, 0.95, 0.98, 0.99, 0.999],
}

rnd_search_bn = RandomizedSearchCV(
    DNNClassifier(random_state=42),
    param_distribs,
    n_iter=50,
    fit_params={
        "X_valid": X_valid1,
        "y_valid": y_valid1,
        "n_epochs": 1000
    },
    random_state=42,
    verbose=2)
rnd_search_bn.fit(X_train1, y_train1)

# Display best paramaters
print("Best paramaters for batch norm", rnd_search_bn.best_params_)
y_pred = rnd_search_bn.predict(X_test1)
acc_bn_best = str("Hypertuned parameters for batch norm {:.2f}".format(
    accuracy_score(y_test1, y_pred)))
model_scores.append(acc_bn_best)

# Adding dropout (reduce overfitting)
dnn_clf_dropout = DNNClassifier(
    activation=leaky_relu(alpha=0.1),
    batch_size=500,
    learning_rate=0.01,
    n_neurons=90,
    random_state=42,
    dropout_rate=0.5)
dnn_clf_dropout.fit(
    X_train1, y_train1, n_epochs=1000, X_valid=X_valid1, y_valid=y_valid1)

# Accuracy for dropout
y_pred = dnn_clf_dropout.predict(X_test1)
acc_dropout = str("Implementing using dropout:".format(
    accuracy_score(y_test1, y_pred)))
model_scores.append(acc_dropout)

# Dropout and randomized search with cross validation
param_distribs = {
    "n_neurons": [10, 30, 50, 70, 90, 100, 120, 140, 160],
    "batch_size": [10, 50, 100, 500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "activation":
    [tf.nn.relu, tf.nn.elu,
     leaky_relu(alpha=0.01),
     leaky_relu(alpha=0.1)],
    "dropout_rate": [0.2, 0.3, 0.4, 0.5, 0.6],
}

# Randomized search using cross validation with dropout implemented
rnd_search_dropout = RandomizedSearchCV(
    DNNClassifier(random_state=42),
    param_distribs,
    n_iter=50,
    fit_params={
        "X_valid": X_valid1,
        "y_valid": y_valid1,
        "n_epochs": 1000
    },
    random_state=42,
    verbose=2)
rnd_search_dropout.fit(X_train1, y_train1)

# The best model_scores paramaeters from randomized search
print("Best results for randomized search including dropout",
      rnd_search_dropout.best_params_)

# Testing the same model_scores with dropout and randomized search parameters
y_pred = rnd_search_dropout.predict(X_test1)
dropout_rnd_search = str("Randomized search with dropout:".format(
    accuracy_score(y_test1, y_pred)))
model_scores.append(dropout_rnd_search)

# Display comparison of model_scores
for model_score in model_scores:
    print(model_score)
