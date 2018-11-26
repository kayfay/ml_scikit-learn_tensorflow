import numpy as np

r = np.arange(24).reshape(6, 4)

#  array([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15],
#         [16, 17, 18, 19],
#         [20, 21, 22, 23]])

# Splitting vertically into 3 equal parts.
r1, r2, r3 = np.vsplit(r, 3)

r1
#  array([[0, 1, 2, 3],
#         [4, 5, 6, 7]])

r2
#  array([[ 8,  9, 10, 11],
#         [12, 13, 14, 15]])

r3
#  array([[16, 17, 18, 19],
#         [20, 21, 22, 23]])

# Splitting horizontally into 2 equal parts.
r4, r5 = np.hsplit(r, 2)

r4
#  array([[ 0,  1],
#         [ 4,  5],
#         [ 8,  9],
#         [12, 13],
#         [16, 17],
#         [20, 21]])

r5
#  array([[ 2,  3],
#         [ 6,  7],
#         [10, 11],
#         [14, 15],
#         [18, 19],
#         [22, 23]])

np.split(r, 3, axis=0)
np.split(r, 3, axis=1)
