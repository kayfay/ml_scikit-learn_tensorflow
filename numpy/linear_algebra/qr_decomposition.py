import numpy as np
from numpy import linalg

# QR decompisition breaks down a Q orthagnonal matrix
# and an R upper triangle matrix
M = np.array([[1, 2, 3], [5, 7, 11], [21, 29, 31]])

q, r = linalg.qr(M)

# r Upper right triangle

#  array([[-21.61018278, -29.89331494, -32.80860727],
#         [  0.        ,   0.62427688,   1.9894538 ],
#         [  0.        ,   0.        ,  -3.26149699]])

# q Orthagnonal matrix

#  array([[-0.04627448,  0.98786672,  0.14824986],
#         [-0.23137241,  0.13377362, -0.96362411],
#         [-0.97176411, -0.07889213,  0.22237479]])

# Orthaogonal unit vectors
q.T.dot(q)  # Identity matrix

#  array([[ 1.00000000e+00, -8.77431704e-17, -6.08691901e-17],
#         [-8.77431704e-17,  1.00000000e+00,  2.11623518e-17],
#         [-6.08691901e-17,  2.11623518e-17,  1.00000000e+00]])

q.dot(r)  # A = QR

#  array([[ 1.,  2.,  3.],
#         [ 5.,  7., 11.],
#         [21., 29., 31.]])
