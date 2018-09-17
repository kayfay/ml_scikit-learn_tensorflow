# Python 2 and 3 compatability
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np
import timeit

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# ML library xgboost import pip3 install --user --upgrade xgboost
try:
    import xgboost
except ImportError as ex:
    print("Error: the xgboost library is not installed.")
    xgboost = None

# Generate some data sets with random noise
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0]**2 + 0.05 * np.random.randn(100)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=41)

# Using XGBoost
if xgboost is not None:
    xgb_reg = xgboost.XGBRegressor(random_state=42)
    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    print("Validation MSE:", val_error)

if xgboost is not None:
    xgb_reg.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=2, verbose=0)
    y_pred = xgb_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    print("Validation MSE w/ early stopping:", val_error)

# inline for notebooks
# %timeit xgboost.XGBRegressor() ...

# timeit's - object cannot be interperted as an integer
#  t = timeit.Timer()

#  try:
    #  t.timeit(xgboost.XGBRegressor().fit(X_train, y_train) if xgboost is not None else None)
    #  t.timeit(GradientBoostingRegressor().fit(X_train, y_train))
#  except:
    #  t.print_exc()
