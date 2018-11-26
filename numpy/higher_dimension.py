import numpy as np

# Higher dimensional i.e., 4D.
c = np.arange(48).reshape(4, 2, 6)

#  array([[[ 0,  1,  2,  3,  4,  5],
#          [ 6,  7,  8,  9, 10, 11]],
#
#         [[12, 13, 14, 15, 16, 17],
#          [18, 19, 20, 21, 22, 23]],
#
#         [[24, 25, 26, 27, 28, 29],
#          [30, 31, 32, 33, 34, 35]],
#
#         [[36, 37, 38, 39, 40, 41],
#          [42, 43, 44, 45, 46, 47]]])

c[2, 1, 4]  # 34 matrix 2, row 1, col 4
c[2, :, 3]  # Matrix 2, all rows, col 3
# array([27, 33])

# All elements omitted coordinates for some axis.
c[2, 1]  # Matrix 2, row 1, all columns
# array([30, 31, 32, 33, 34, 35])

# Ellipsis (...) include all non-specified axes.
c[2, ...]  # Matrix 2, all row / col, also c[2, :, :]
#  array([[24, 25, 26, 27, 28, 29],
#         [30, 31, 32, 33, 34, 35]])

c[2, 1, ...]  # Matrix 2, row 1, all columns, also c[2, 1, :]
# array([30, 31, 32, 33, 34, 35])

c[2, ..., 3]  # Matrix 2, all rows, col 3, also c[2, :, 3]
# array([27, 33])

c[..., 3]  # All matrices, all rows, column 3, also c[:, :, 3]

#  array([[ 3,  9],
#         [15, 21],
#         [27, 33],
#         [39, 45]])
