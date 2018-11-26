import numpy as np
from numpy import linalg

M = np.array([[1, 2, 3], [5, 7, 11], [21, 29, 31]])

# Compute the  eigenvectors (characteristic vectors)
# and eigenvalues (characteristic values) of a square matrix
eigenvalues, eigenvectors = linalg.eig(M)

eigenvalues
# array([42.26600592, -0.35798416, -2.90802176])

eigenvectors

#  array([[-0.08381182, -0.76283526, -0.18913107],
#         [-0.3075286 ,  0.64133975, -0.6853186 ],
#         [-0.94784057, -0.08225377,  0.70325518]])

M.dot(eigenvectors) - eigenvalues * eigenvectors  # M.v - lambda * v = 0

#  array([[ 6.66133815e-15,  1.66533454e-15, -3.10862447e-15],
#         [ 7.10542736e-15,  5.16253706e-15, -5.32907052e-15],
#         [ 3.55271368e-14,  4.94743135e-15, -9.76996262e-15]])
