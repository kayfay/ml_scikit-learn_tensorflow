import numpy as np

# Multidimensional arrays
c = np.arange(24).reshape(2, 3, 4)

#  array([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])

# Iterating over multidimensional arrays.
for m in c:  # As a Python iterator
    print("Item:")
    print(m)

#  Item:
#  [[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#  Item:
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]

for i in range(len(c)):  # As iteration an ndarray
    print("Item:")
    print(c[i])

#  Item:
#  [[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#  Item:
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]

# To flatten the matrix and iterate on only elements.
for i in c.flat:
    print("Item:", i)

#  Item: 1
#  Item: 2
#  Item: 3
#
#  ...
#
#  Item: 21
#  Item: 22
#  Item: 23
