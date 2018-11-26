import numpy as np

a = np.array([1, 5, 3, 19, 13, 7, 3])
a[3]  # 19

a[2:5]  # array([ 3, 19, 13])

a[2:-1]  # array([ 3, 19, 13,  7])

a[:2]  # array([1, 5])

a[::-1]  # array([ 3,  7, 13, 19,  3,  5,  1])

a[3] = 999  # array([  1,   5,   3, 999,  13,   7,   3])

a[2:5] = [997, 998, 999]  # array([  1,   5, 997, 998, 999,   7,   3])

# Regular python arrays and ndarays act differently, from broadcasting.
a[2:5] = -1
a  # array([ 1,  5, -1, -1, -1,  7,  3])

# Except
lc = list(a)
list(a)  # [1, 5, 997, 998, 999, 7, 3]
lc[2:5] = -1  # TypeError: can only assign an iterable

try:
    a[2:5] = [1, 2, 3, 4, 5, 6]  # too long
except ValueError as e:
    print(e)
# cannot copy sequence with size 6 to array axis with dimension 3

try:
    del a[2:5]
except ValueError as e:
    print(e)
# cannot delete array elements

# Slicing are pointers / views on the same data.
a_slice = a[2:6]
a_slice[1] = 1000
a
# array([   1,    5,   -1, 1000,   -1,    7,    3])

# Modifing the original array modifies the slice.
a[3] = 2000
a_slice  # array([  -1, 2000,   -1,    7])

# Copying data using the copy method.
another_slice = a[2:6].copy()
another_slice[1] = 3000
a  # array([   1,    5,   -1, 2000,   -1,    7,    3])

a[3] = 4000
another_slice  # array([  -1, 3000,   -1,    7])

# Multi-dimensional arrays use index, slice, axis
b = np.arange(48).reshape(4, 12)

#  array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
#         [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
#         [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
#         [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]])

b[1, 2]  # row 1 col 2
b[1, :]  # row 1 all col
b[:, 1]  # all rows col 1

# 1D array vs 2D array
b[1, :]  # (12,)
b[1:2, :]  # (1, 12)

# Fancy indexing a list of indices
b[(0, 2), 2:5]  # rows 0 and 2, columns starting 2 upto 5

#  array([[ 2,  3,  4],
#         [26, 27, 28]])

b[:, (-1, 2, -1)]  # all rows, columns -1, 2, and -1
#  array([[11,  2, 11],
#         [23, 14, 23],
#         [35, 26, 35],
#         [47, 38, 47]])

b[(-1, 2, -1, 2), (5, 9, 1, 9)]  # row / col coordinates
# array([41, 33, 37, 33])
