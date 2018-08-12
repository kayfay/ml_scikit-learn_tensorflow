### Setup Environment for Py 2 and Py 3
from __future__ import division, print_function, unicode_literals

# Import packages
import numpy as np
import os
import tarfile
from six.moves import urllib
import pandas as pd
import scipy
from scipy import stats

# Import preprocessing and machine learning models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from future_encoders import OneHotEncoder
# from category_encoders import OneHotEncoder  # future_encoders replac ln 128
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

## Data Info

# Declare data filenames
DATAFILE_NAME = "housing.tgz"
DATAFILE_FORMAT = "housing.csv"

# Declare directory and network location of stored data
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
DATA_PATH = os.path.join("datasets", "housing")
DATA_URL = DOWNLOAD_ROOT + "datasets/housing/" + DATAFILE_NAME


# Define fetch data function
def fetch_data(data_url=DATA_URL, data_path=DATA_PATH,
               datafile_name=DATAFILE_NAME):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    tgz_path = os.path.join(data_path, datafile_name)
    urllib.request.urlretrieve(data_url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path=data_path)
    data_tgz.close()


# Define load data function
def load_data(path=DATA_PATH, datafile_format=DATAFILE_FORMAT):
    csv_path = os.path.join(path, datafile_format)
    return pd.read_csv(csv_path)


## Define class for feature engineering

# Declare feature indexes from dataframe
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


# Custom class transformer combining attributes
class CombinedAttributesAddr(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, rooms_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# Custom class for selecting numerical or catagorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


## Preprocess data

# Get and Set up Data

fetch_data()
housing = load_data()

# Build a median income catagory to use for labels w/ 5 catagories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

# Create training and test sets, stratify and shuffle the data sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Split feature data from labeled data
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Create attributes for indexing
num_attribs = list(housing.drop('ocean_proximity', axis=1))
cat_attribs = ['ocean_proximity']

# Build pipeline for pre-processing
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAddr()),
        ('std_scaler', StandardScaler()),
     ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),  #Rem sparse see ln 19
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])

# Process data with pipeline
housing_prepared = full_pipeline.fit_transform(housing)

## Train some Models

# Declare instance for a support vector machine
svm_reg = SVR()

# Declare paramater grid for RandomizedSearchCV
param_distributions = {
        # Declare C range and gamma range for search paramaters
        'C': scipy.stats.expon(scale=100),
        'gamma': scipy.stats.expon(scale=.1),
        'kernel': ['rbf'],
    }

# Declare subsets for selecting good hyperparameters
subset = housing.iloc[:100]
subset_labels = housing.iloc[:100]

# Perform RandomizedSearch using SVM regression
rnd_search = RandomizedSearchCV(svm_reg, param_distributions, n_iter=10, cv=5,
                                scoring='neg_mean_squared_error',
                                random_state=42)

# Select for full test or quick test
# rnd_search.fit(housing_prepared, housing_labels) # full data set
rnd_search.fit(subset, subset_labels)  # quick test

## Determine the top best features

# Store the best features from the best estimator paramaters
feature_importances = rnd_search.best_estimator_.feature_importances_

# Create a reference list
extra_attribs = ["rooms_per_household", "population_per_household",
                 "bedrooms_per_room"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

# Declare final model
final_model = rnd_search.best_estimator_

# Declare test sets
X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

# Process data with pipeline
X_test_prepared = full_pipeline.transform(X_test)

# Predict from model and declare as variable
final_predictions = final_model.predict(X_test_prepared)

# Create a 95% confidence interval
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
m = len(squared_errors)

interval_values = np.sqrt(stats.t.interval(confidence, m - 1,
                            loc=np.mean(squared_errors),
                            scale=stats.sem(squared_errors)))

# Create metrics and display
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
top5_features = sorted(zip(feature_importances, attributes),
                       reverse=True)[:5]


print("RMSE 95% confidence interval: ", interval_values)
print("The best classifier is: ", rnd_search.best_estimator_)
print("A 95% prediction confidence interval")
print("Root Mean Square Error from predictions: ", final_rmse)
print("The top five features.\n", top5_features)
