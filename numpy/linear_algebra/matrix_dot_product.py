import numpy as np

# Dot product by taking a matrix of M by N
M = np.arange(10).reshape(2, 5)
N = np.arange(15).reshape(5, 3)

# M

#  array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# N

#  array([[ 0,  1,  2],
#         [ 3,  4,  5],
#         [ 6,  7,  8],
#         [ 9, 10, 11],
#         [12, 13, 14]])

M.dot(N)

#  array([[ 90, 100, 110],
#         [240, 275, 310]])
