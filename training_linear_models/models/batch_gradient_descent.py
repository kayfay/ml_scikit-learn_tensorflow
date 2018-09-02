# Example of bath gradient descent
# Common Imports
import numpy as np

# Declare varables
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

# Declare paramaters
eta = 0.1  # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1)  # random initializations

#  # Gradient Descent step

# Batch Gradient Descent algorithm
for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print("Resulting estimated coefficient/theta parameter weight: \n", theta)
