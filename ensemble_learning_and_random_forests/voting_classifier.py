# Ensemble classifier using hard/soft voting composing of a
# multilayer perceptron, forests, and a linear support vectors.
# Create a stacking ensemble blending the ensemble predictions

# Common Imports
import numpy as np

# ML Imports

# Data Imports
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
# Classifier Imports
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
# Metric Imports
from sklearn.metrics import accuracy_score


# Declare Functions
def display_estimators(clf_list, score_list):
    for i, j, in zip(clf_list, score_list):
        print("Estimator:", i)
        print("Estimator:", j)


# Create training, test, and validation sets
mnist = fetch_mldata("MNIST original")

X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)

# Instantiate models
random_forest_clf = RandomForestClassifier(random_state=42)
extra_tree_clf = ExtraTreesClassifier(random_state=42)
svm_clf = LinearSVC(random_state=42)
mlp_clf = MLPClassifier(random_state=42)

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_tree_clf", extra_tree_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]

voting_clf = VotingClassifier(named_estimators)

# Fit the Voting classifier
voting_clf.fit(X_train, y_train)

# Train models
estimators = [random_forest_clf, extra_tree_clf, svm_clf, mlp_clf]
for estimator in estimators:
    estimator.fit(X_train, y_train)

scores = [estimator.score(X_val, y_val) for estimator in estimators]
display_estimators(estimators, scores)

# Remove SVC
print("Without the LinearSVC")
voting_clf.set_params(svm_clf=None)

scores = [estimator.score(X_val, y_val) for estimator in estimators]
display_estimators(estimators, scores)

del voting_clf.estimators_[2]

# Set show soft v hard voting classification
hard = voting_clf.score(X_val, y_val)
voting_clf.voting = "soft"
soft = voting_clf.score(X_val, y_val)

print("Hard voting score compared to Softvoting score")
print("Hard: {}, Soft: {}".format(hard, soft))

# Generate predictions using the estimators for a stacking ensemble
X_val_predictions = np.empty((len(X_val), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X_val)

rnd_forest_blender = RandomForestClassifier(
    n_estimators=200, oob_score=True, random_state=42)
rnd_forest_blender.fit(X_val_predictions, y_val)

# Evaluate on the test set
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)

y_pred = rnd_forest_blender.predict(X_test_predictions)

# Stacking prediction ensembles
print("Stacking ensemble predictions")
print("OOB CV Training Accuracy score:{}".format(
    rnd_forest_blender.oob_score_))
print("Test accuracy score:{}".format(accuracy_score(y_test, y_pred)))
