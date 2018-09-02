# Batch Gardient Descent with early stopping for softmax regression
# Common Imports
import numpy as np

# Dataset imports
from sklearn import datasets

# Graphic Imports / Config
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define functions


def to_one_hot(y):
    # i.e. [1, 0, 0] for [0] or [0, 1, 0] for [1]
    m = len(y)
    n_classes = y.max() + 1
    Y_one_hot = np.zeros((m, n_classes))
    Y_one_hot[np.arange(m), y] = 1
    return Y_one_hot


def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exp_sums


# Import and set up data
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = iris["target"]

X_with_bias = np.c_[np.ones([len(X), 1]), X]

# Split datasets
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)  # Add a bias column

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

# Randomize the data
rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

# Convert y sets into one_hot_vectors
Y_train_one_hot = to_one_hot(y_train)
Y_val_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)

# Softmax computation
# Define hyperparameters
eta = 0.1  # learning rate
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7  # added error
alpha = 0.1
best_loss = np.infty

# Define num inputs / outputs for random Theta initialization
n_inputs = X_train.shape[1]  # count feats as 1b+2(w/l)
n_outputs = len(np.unique(y_train))  # count classes
Theta = np.random.randn(n_inputs, n_outputs)

# Execute Softmax computation
print("Iteration, Cross Entropy, Accuracy Score\n")

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)  # sigmoid function
    Y_proba = softmax(logits)  # estimates

    # Cost/loss function, cross entropy
    loss = -np.mean(
        np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1 / 2 * np.sum(np.square(Theta[1:]))
    loss = loss + alpha * l2_loss  # added regularization/hyperparameter

    # Display information about process

    # Compute validation set RMSE
    logits_val = X_valid.dot(Theta)
    Y_proba_val = softmax(logits_val)
    loss = -np.mean(
        np.sum(Y_val_one_hot * np.log(Y_proba_val + epsilon), axis=1))
    l2_loss = 1 / 2 * np.sum(np.square(Theta[1:]))
    loss = loss + alpha * l2_loss
    y_predict = np.argmax(Y_proba_val, axis=1)
    accuracy_score = np.mean(y_predict == y_valid)

    # Gradient function: 1/m { sum ( p - y ) * x } + a * theta
    error = Y_proba - Y_train_one_hot
    gradients = 1 / m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]),
                                                     alpha * Theta[1:]]
    Theta = Theta - eta * gradients  # weight steps, learning rate

    # Display
    if iteration % 500 == 0:  # display status updates
        print(iteration, loss, accuracy_score)
    if loss < best_loss:
        best_loss = loss
    else:
        print(iteration - 1, best_loss, accuracy_score)
        print("\nBest measure, implementing early stopping\n", iteration, loss,
              accuracy_score)
        break

# Create axis grid for X matrix map
x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500).reshape(-1, 1),
    np.linspace(0, 3.5, 200).reshape(-1, 1),
)

# Map X to euclidian space
X_plot = np.c_[x0.ravel(), x1.ravel()]
X_plot_with_bias = np.c_[np.ones([len(X_plot), 1]), X_plot]

# Calculate estimates
logits = X_plot_with_bias.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

# Contor vectors, z axis
z_c_label = Y_proba[:, 1].reshape(x0.shape)
z_c = y_predict.reshape(x0.shape)

# Plot data points
plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")

# Colormap
custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

# Plot contour lines
plt.contourf(x0, x1, z_c, cmap=custom_cmap)
contour = plt.contour(x0, x1, z_c_label, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()
