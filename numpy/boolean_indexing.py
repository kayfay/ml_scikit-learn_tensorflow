import numpy as np

b = np.arange(48).reshape(4, 12)

#  array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
#         [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
#         [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
#         [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]])

# Specifying indices on an axis with boolean values.
rows_on = np.array([True, False, True, False])
b[rows_on, :] # Rows 0 and 2, all columns, also b[(0, 2), :]

#  array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
#         [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]])

cols_on = np.array([False, True, False] * 4)
b[:, cols_on] # All rows, col 1, 4, 7, 10

#  array([[ 1,  4,  7, 10],
#         [13, 16, 19, 22],
#         [25, 28, 31, 34],
#         [37, 40, 43, 46]])

# Boolean indexing on multiple axes.
b[np.ix_(rows_on, cols_on)] # ix rows[T,F,T,F,] by cols[F,T,F] * 4

#  array([[ 1,  4,  7, 10],
#         [25, 28, 31, 34]])

np.ix_(rows_on, cols_on) # Index of rows by columns by booleans

#  (array([[0], [2]]), array([[ 1,  4,  7, 10]]))

# Using considtional operators.
b[b % 3 == 1]

# array([ 1,  4,  7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46])
