# Least Absolute Shrinkage and Selection Operator Regression
# Lasso Regression

# Common Imports
import numpy as np

# ML Imports
from sklearn.linear_model import Lasso, SGDRegressor

# Declare variables
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5

# Compare
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y.ravel())
sgd_reg = SGDRegressor(max_iter=5, penalty="l1", random_state=42)
sgd_reg.fit(X, y.ravel())
print("Lasso, SGD: \n", lasso_reg.predict([[1.5]]), sgd_reg.predict([[1.5]]))
