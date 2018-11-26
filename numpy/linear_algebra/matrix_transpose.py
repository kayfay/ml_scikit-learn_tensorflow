import numpy as np

m1 = np.arange(10).reshape(2, 5)

#  array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# Transpose a matrix.
m1.T

#  array([[0, 5],
#         [1, 6],
#         [2, 7],
#         [3, 8],
#         [4, 9]])

# Rank 1 v Rank 2 matrix transpose
m2 = np.arange(5)
# array([[0, 1, 2, 3, 4]])
m2.shape  # (5,) Rank 1

m2 = m2.reshape(1, 5)
# array([[0, 1, 2, 3, 4]])
m2.shape  # (1, 5) Rank 2

# Using the matrix transpose requires Rank 2, vector v matrix.
m2.T

#  array([[0],
#         [1],
#         [2],
#         [3],
#         [4]])
