# Common Imports
import numpy as np
import matplotlib.pyplot as plt

# ML Imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# quadratic expression
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Instantiate the the preprocessor
# ex. [a, b], 2 degree: [1, a, b, a^2, ab, b^2]
poly_features = PolynomialFeatures(degree=2, include_bias=False)

# Transform the dataset
# ex create polynomial combinations of X to each 2 degree
X_poly = poly_features.fit_transform(X)

# Fit linear regression model with polynomials as a quadratic expression

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# plot
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
plt.show()
