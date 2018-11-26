import numpy as np

M = np.array([[1, 2, 3], [5, 7, 11], [21, 29, 31]])

#  array([[ 1,  2,  3],
#         [ 5,  7, 11],
#         [21, 29, 31]])

# The diagnoal from row 1 col 1 to row 3 col 3
# Ecludian distance from M[1, 1] to  M[-1, -1]
np.diag(M)

# array([ 1,  7, 31])

# Calculate the sum along the diagnoal
np.trace(M)  # 39 as is np.diag.sum()
