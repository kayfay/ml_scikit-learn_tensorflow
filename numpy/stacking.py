import numpy as np

# Stacking arrays
q1 = np.full((3, 4), 1.0)

#  array([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]])

q2 = np.full((4, 4), 2.0)

#  array([[2., 2., 2., 2.],
#         [2., 2., 2., 2.],
#         [2., 2., 2., 2.],
#         [2., 2., 2., 2.]])

q3 = np.full((3, 4), 3.0)

#  array([[3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.]])

# Vertical stacking ques/queries based on equal dimensions.
q4 = np.vstack((q1, q2, q3))

q4.shape  # (10, 4)

#  array([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [2., 2., 2., 2.],
#         [2., 2., 2., 2.],
#         [2., 2., 2., 2.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.],
#         [3., 3., 3., 3.]])

# Horizontal stacking ques/queries based on equal dimensions.
q5 = np.hstack((q1, q3))

#  array([[1., 1., 1., 1., 3., 3., 3., 3.],
#         [1., 1., 1., 1., 3., 3., 3., 3.],
#         [1., 1., 1., 1., 3., 3., 3., 3.]])

# Non-equal dimensions, q1 (3, 4), q2 (4, 4), q3 (3, 4) cannot be stacked.
try:
    q5 = np.hstack((q1, q2, q3))
except ValueError as e:
    print(e)

# all the input array dimensions except
# for the concatenation axis must match exactly

# Concatenate can also perform vstack and hstack
q7 = np.concatenate((q1, q2, q3), axis=0)
q7 = np.concatenate((q1, q3), axis=1)

# Stacks arrays along a new axis with the same shape creating new dimensions.
q8 = np.stack((q1, q3))
q8.shape  # (2, 3, 4)

#  array([[[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.]],
#
#         [[3., 3., 3., 3.],
#          [3., 3., 3., 3.],
#          [3., 3., 3., 3.]]])
