import numpy as np

t = np.arange(24).reshape(4, 2, 3)

#  array([[[ 0,  1,  2],
#          [ 3,  4,  5]],
#
#         [[ 6,  7,  8],
#          [ 9, 10, 11]],
#
#         [[12, 13, 14],
#          [15, 16, 17]],
#
#         [[18, 19, 20],
#          [21, 22, 23]]])

# Transpose creates a new pointer/view with axes permuted d, h, w
# depth, width, height.
t1 = t.transpose((1, 2, 0))  # Axis 0, 1, 2 reordered to 1, 2, 0.

t1.shape  # (2, 3, 4)

#  array([[[ 0,  6, 12, 18],
#          [ 1,  7, 13, 19],
#          [ 2,  8, 14, 20]],
#
#         [[ 3,  9, 15, 21],
#          [ 4, 10, 16, 22],
#          [ 5, 11, 17, 23]]])

np.ndarray()[0, 1, 2, 3].transpose()
[0, 1, 2, 3]

t = np.arange(24).reshape(4, 2, 3)

#  array([[[ 0,  1,  2],
#          [ 3,  4,  5]],
#
#         [[ 6,  7,  8],
#          [ 9, 10, 11]],
#
#         [[12, 13, 14],
#          [15, 16, 17]],
#
#         [[18, 19, 20],
#          [21, 22, 23]]])

# Transpose creates a new pointer/view with axes permuted d, h, w
# depth, width, height.
t1 = t.transpose((1, 2, 0))  # Axis 0, 1, 2 reordered to 1, 2, 0.

t1.shape  # (2, 3, 4)

#  array([[[ 0,  6, 12, 18],
#          [ 1,  7, 13, 19],
#          [ 2,  8, 14, 20]],
#
#         [[ 3,  9, 15, 21],
#          [ 4, 10, 16, 22],
#          [ 5, 11, 17, 23]]])

# Reverse order of dimensions by default.
t2 = t.transpose()

t2.shape  # (3, 2, 4)

#  array([[[ 0,  6, 12, 18],
#          [ 3,  9, 15, 21]],
#
#         [[ 1,  7, 13, 19],
#          [ 4, 10, 16, 22]],
#
#         [[ 2,  8, 14, 20],
#          [ 5, 11, 17, 23]]])

# Swap depth and height.
t3 = t.swapaxes(0, 1)

t3.shape  # (2, 4, 3)

#  array([[[ 0,  1,  2],
#          [ 6,  7,  8],
#          [12, 13, 14],
#          [18, 19, 20]],
#
#         [[ 3,  4,  5],
#          [ 9, 10, 11],
#          [15, 16, 17],
#          [21, 22, 23]]])
