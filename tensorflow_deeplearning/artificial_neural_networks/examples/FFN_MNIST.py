"""
Using a Feed Forward Neural network with the MNIST images dataset
Predict results using a DNN, the SKlearn estimators
"""

import tensorflow as tf
import numpy as np

# Get Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# Define feature columns
feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]

# Instantiate a classifier
dnn_clf = tf.estimator.DNNClassifier(
    hidden_units=[300, 100], n_classes=10, feature_columns=feature_cols)

# Instantiate an input function classifier
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)

# Train a classifier
dnn_clf.train(input_fn=input_fn)

#  Instantiate a classifier
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)

# Display Results
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
y_pred = list(y_pred_iter)
y_pred[0]

print(eval_results, y_pred[0])
