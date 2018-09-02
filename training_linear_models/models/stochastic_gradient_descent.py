# Common Imports
import numpy as np

# ML imports
from sklearn.linear_model import SGDRegressor

# Declare varables
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

# Declare parameters
eta = 0.1  # learning rate
n_iterations = 1000
m = 100

n_epochs = 50  # iterate through the gradient descent per epoch

t0, t1 = 5, 50  # learning schedule hyper parameters

theta = np.random.randn(2, 1)  # random initializations


# Declare functions
def learning_schedule(t):
    # A simulated annealing function to gradually reduce
    # the gradient descent to converge on a global minima
    return t0 / (t + t1)


# Implementing Stochastic Gradient Descent
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

# performing stochastic gradient descent linear regression
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

print("Estimated weights/coeficients: \n", theta)
print("Estimated by SGDRegressor: \n", sgd_reg.intercept_, sgd_reg.coef_)
