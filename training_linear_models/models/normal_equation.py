# An example of using the normal equation to perform linear predictions

# Scientific package and numper imports
import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve

# Plotting imports
import matplotlib.pyplot as plt

# ML imports
from sklearn.linear_model import LinearRegression

# Declare variables
x = [[1., 2.], [3., 4.]]
x_t = np.transpose(x)
y = np.array([1, 0])

X_1 = 2 * np.random.rand(100, 1)
y_1 = 4 + 3 * X_1 + np.random.randn(100, 1)  # y = 4 +3x1 + Gaussin noise
X_b = np.c_[np.ones((100, 1)), X_1]

# Normal equation
# theta_hat = (x_transpose • x )^-1 • x_transpose • y
theta_hat = np.dot(np.dot(inv(np.dot(x_t, x)), x_t), y)

# Or
x_t_x = np.dot(x_t, x)
x_t_y = np.dot(x_t, y)

theta_hat_simple = np.dot(inv(x_t_x), x_t_y)

# Or solve a linear matrix equation
# solve a system of linear scalar equations
print("Normal Equation: ", solve(x, y), "=", theta_hat, "=", theta_hat_simple)

# Theta paramaeter estimates
theta_best = inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_1)
print("With y=4+3x1 + noise: \n", theta_best,
      "\n The estimates are close to 4 and 3\n")

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print("A set of predictions: ", y_predict)

# plt.plot(X_new, y_predict, 'r-')
# plt.plot(X_1, y_1, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X_1, y_1)
print("In Sklearn:", lin_reg.intercept_, lin_reg.coef_)
print("Predict: ", lin_reg.predict(X_new))
