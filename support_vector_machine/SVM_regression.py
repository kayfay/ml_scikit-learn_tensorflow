# Using a SVM for regression tasks on the california housing data
# Python 2 and 3 support
from __future__ import division, unicode_literals, print_function

# Common Imports
import numpy as np

# ML Imports
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Stats Imports
from scipy.stats import reciprocal, uniform

# Set up data
housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Base Model
lin_svr = LinearSVR(random_state=42)
lin_svr.fit(X_train_scaled, y_train)

# Predict and compute performance metrics
y_pred = lin_svr.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(
    mse)  # .97 with average housing price this is a high error of 10k+-

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(
    SVR(), param_distributions, n_iter=10, verbose=2, random_state=42)

rnd_search_cv.fit(X_train_scaled, y_train)
y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
rmse  # training error

y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse  # test error .57
print("Error rate for predictions: ", rmse)
