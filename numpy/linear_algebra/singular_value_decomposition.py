import numpy as np
from numpy import linalg

# Matrix factorization is composed of three elements

#  1. U Unitary matrix
#  m by m complex square matrix
#  conjugate transpose U* complex conjugate,
#  a conjugate of imaginary values (e.g., a - bi, a + bi).

#  2. Singular values
#  square roots of eigenvalues of the
#  non-negative self-adjoint operator

#  3. V n by n real or complex unitary matrix
#  where U are right-singular vectors
#  and V are left-singular vectors

M = np.array([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0],
              [0, 2, 0, 0, 0]])

U, S_diag, V = linalg.svd(M)

#  U

#  array([[ 0.,  1.,  0.,  0.],
#         [ 1.,  0.,  0.,  0.],
#         [ 0.,  0.,  0., -1.],
#         [ 0.,  0.,  1.,  0.]])

#  S_diag

#  array([3.        , 2.23606798, 2.        , 0.        ])

# Sigma matrix
S = np.hstack((np.diag(S_diag), np.zeros((4, 1))))

#  array([[3.        , 0.        , 0.        , 0.        , 0.        ],
#         [0.        , 2.23606798, 0.        , 0.        , 0.        ],
#         [0.        , 0.        , 2.        , 0.        , 0.        ],
#         [0.        , 0.        , 0.        , 0.        , 0.        ]])

#  V

#  array([[-0.        ,  0.        ,  1.        , -0.        ,  0.        ],
#         [ 0.4472136 ,  0.        ,  0.        ,  0.        ,  0.89442719],
#         [-0.        ,  1.        ,  0.        , -0.        ,  0.        ],
#         [ 0.        ,  0.        ,  0.        ,  1.        ,  0.        ],
#         [-0.89442719,  0.        ,  0.        ,  0.        ,  0.4472136 ]])

U.dot(S).dot(V)  # M = U.S.V

#  array([[1., 0., 0., 0., 2.],
#         [0., 0., 3., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 2., 0., 0., 0.]])
