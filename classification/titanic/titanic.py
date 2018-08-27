# To support both python 2 & 3
from __future__ import division, print_function, unicode_literals

# Common imports
import os
import numpy as np
import pandas as pd

# Machine learning imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from future_encoders import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


# matplotlib imports
# import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Path variables
TITANIC_PATH = os.path.join("datasets", "titanic")

# Save fig directory
PROJECT_ROOT_DIR = "."
DIRECTORY = "titanic"

# Set seed value
np.random.seed(42)

# Custom Functions


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", DIRECTORY, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def load_data(filename, path):
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path)


def col_sum(data, new, col1, col2):
    data[str(new)] = data[str(col1)] + data[str(col2)]
    return data.drop([str(col1), str(col2)], axis=1)


def plot_boxplots(scores, savename):
    plt.figure(figsize=(8, 4))
    for i in range(1, scores.size()):
        plt.plot([i]*10, scores.values()[i], ".")
        plt.boxplot(scores.values(), labels=scores.keys())
    plt.ylabel("accuracy", fontsize=14)
    save_fig(savename)


# Custom classes


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
            [X[c].value_counts().index[0] for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# Declare train/test data
train_data = load_data("train.csv", TITANIC_PATH)
test_data = load_data("test.csv", TITANIC_PATH)

# Use in Jupyter or similar
# train_data.head()
# 891 entries, 12 features, survived is target 0/1

# train_data.info()
# Name/Ticket -null_val
# Age/Cabin/Embarked -null_val Cabin -77%null_val

# train_data.describe()
# mean 38% survival, 32.20 fare, 30 age

# Inspect catagorical attributes
# train_data["Survived"].value_coutns()
# train_data["Pclass"].value_counts() # passenger class
# train_data["Sex"].value_counts()
# train_data["Embarked"].value_counts()
# where from Cherboug, Queenstown, Southampton

# Feature enginering Family members from Siblings, Spouces, and Parents
train_data = col_sum(train_data, 'FamMems', 'SibSp', 'Parch')
test_data = col_sum(test_data, 'FamMems', 'SibSp', 'Parch')


# Build pipeline
imputer = Imputer(strategy="median")

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(
        ["Age", "FamMems", "Fare"])),
    ("imputer", Imputer(strategy="median")),
])

num_pipeline.fit_transform(train_data)

cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False)),
])

cat_pipeline.fit_transform(train_data)

preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

X_train = preprocess_pipeline.fit_transform(train_data)
# X_train # to view uncomment

# Make y training set
y_train = train_data["Survived"]

# Declare model and fit model
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# Run preprocessor and predictor
X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)

# Generate base results
print("Base Estimations")
# Create cross validation scores
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print("Support Vector Classifier: ", svm_scores.mean())

# Create cross val for second model
forest_clf = RandomForestClassifier(random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print("RandomForestClassifier: ", forest_scores.mean())

# Create cross val for third model
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                    max_depth=1, random_state=42)
gb_scores = cross_val_score(gb_clf, X_train, y_train, cv=10)
print("GradientBoostingClassifier: ", gb_scores.mean())

# Create cross val for fourth model
nn_clf = MLPClassifier()
nn_scores = cross_val_score(nn_clf, X_train, y_train, cv=10)
print("MLPClassifier: ", nn_scores.mean())

# Create cross val for fifth model
log_clf = LogisticRegression()
log_scores = cross_val_score(log_clf, X_train, y_train, cv=10)
print("LogisticRegression: ", log_scores.mean())

# Create box plots
# plt.figure(figsize=(8, 4)) # replace with function
# plt.plot([1]*10, svm_scores, ".")
# plt.plot([2]*10, forest_scores, ".")
# plt.plot([3]*10, gb_scores, ".")
# plt.plot([4]*10, nn_scores, ".")
# plt.plot([5]*10, log_scores, ".")
# plt.boxplot([svm_scores, forest_scores, gb_scores, nn_scores, log_scores],
#             labels=("SVM", "Random Forest", "GradBoost", "NeuralNet", "log"))
# plt.ylabel("accuracy", fontsize=14)
# save_fig('box_plots')

base_dict_scores = {
    'SVM': svm_scores,
    'Random Forest': forest_scores,
    'GradBoost': gb_scores,
    'NeuralNet': nn_scores,
    'log': log_scores
}

plot_boxplots(base_dict_scores, 'base_estimations_box_plots')

# Hypertune parameters, REDUCE ARGUMENTS BEFORE TEST
svm_param_grid = {
    'C': [1, 10, 100, 200, 300, 1000],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': [0.001, 0.01, 0.1, 1]
}

forest_param_grid = {
    'n_estimators': [100, 300],
    'criterion': ['gini', 'entropy'],
    'max_features': [1, 3, 10, 'sqrt', 'log2'],
    'max_depth': ['None', 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [0.1, 0.5, 1, 3, 10],
    'bootstrap': [True, False]
}

gb_param_grid = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.1, 0.5, 0.01],
    'max_depth': [4, 8],
    'min_samples_leaf': [100, 150],
    'max_features': [0.3, 0.1]
}

grid_search_svm = GridSearchCV(svm_clf, svm_param_grid, "accuracy",
                               n_jobs=4, cv=3, verbose=1)

grid_search_forest = GridSearchCV(forest_clf, forest_param_grid, "accuracy",
                                  n_jobs=4, cv=3, verbose=1)

grid_search_gb = GridSearchCV(gb_clf, gb_param_grid, "accuracy", n_jobs=4,
                              cv=3, verbose=1)

h_tuned_dict = {
    'SVC': grid_search_svm,
    'RandomForest': grid_search_forest,
    'GradBoost': grid_search_gb}

results = []

for m, gs in h_tuned_dict.items():
    gs.fit(X_train, y_train)
    results.append(m, ": ", gs.best_estimator_, "\n")


plot_boxplots(h_tuned_dict, 'hypertuned_box_plots')

print(results)
