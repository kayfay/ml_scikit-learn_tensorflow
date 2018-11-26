import numpy as np
import numpy.linalg as linalg

# To create an identity matrix.
np.eye(3)

#  array([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])

# The product of a matrix by it's inverse returns the identity matrix.
M = np.array([[1, 2, 3], [5, 7, 11], [21, 29, 31]])

M.dot(linalg.inv(M))

#  array([[ 1.00000000e+00, -1.66533454e-16,  0.00000000e+00],
#         [ 6.31439345e-16,  1.00000000e+00, -1.38777878e-16],
#         [ 5.21110932e-15, -2.38697950e-15,  1.00000000e+00]])
