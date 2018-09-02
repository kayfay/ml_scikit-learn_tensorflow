# Compare Ridge regression to Stochastic Gradient Descent

# Common Imports
import numpy as np

# ML Imports
from sklearn.linear_model import Ridge, SGDRegressor

# Declare variables
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5

# Compare
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
# Stochastic Average GD
# ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
ridge_reg.fit(X, y)
sgd_reg = SGDRegressor(max_iter=5, penalty="l2", random_state=42)
sgd_reg.fit(X, y)
print("Ridge, SGD: \n", ridge_reg.predict([[1.5]]), sgd_reg.predict([[1.5]]))
