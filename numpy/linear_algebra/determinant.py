import numpy as np
from numpy import linalg

M = np.array([[1, 2, 3], [5, 7, 11], [21, 29, 31]])

# The determinant is the
# top start to end portions of a matrix
# geometrically viewed as a scaling factor.

linalg.det(M)  # 43.99999999999997
