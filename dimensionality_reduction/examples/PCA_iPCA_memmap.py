# Python 2 and 3 compatibly
from __future__ import division, print_function, unicode_literals

# Common Imports
import os
import numpy as np

# ML Imports
from six.moves import urllib
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, IncrementalPCA

# Graph Imports
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12


# Data
mnist = fetch_mldata('MNIST original')
X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Number of components for expected variance of explicit

# pca = PCA()
# pca.fit(X_train)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmaxc(cumsum >= 0.95) + 1

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)

# pca = PCA(n_components=154)
# X_reduced = pca.fit_transform(X_train)

X_recovered = pca.inverse_transform(X_reduced)

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    print(".", end="")
    inc_pca.partial_fit(X_batch)

X_reduced_inc_pca = inc_pca.transform(X_train)

# An alternative to IncrementalPCA
# Create memmap file for PCA large dataset processing
# filename = "my_mnist.data"
# m, n = X_train.shape
# X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m,n))
# X_mm[:] = X_train
# 
# del X_mm # save the oject to disk
# 
# # Load memmap file for training
# X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
# batch_size = m //n_batches
# inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
# inc_pca.fit(X_mm)
# 
# rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
# X_reduced = rnd_pca.fit_transform(X_train)

print("Number of PCA Components: {}".format(pca.n_components_))
print("Explained variance ratio: {}".format(
    np.sum(pca.explained_variance_ratio_)))
print("Test incremental to regular PCA, close comparison:",
      np.allclose(pca.mean_, inc_pca.mean_))
print("Test incremental to regular PCA, same:",
      np.allclose(X_reduced_inc_pca, X_reduced_inc_pca))
