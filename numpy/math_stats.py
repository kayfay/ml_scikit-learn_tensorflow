import numpy as np

# Mathematical and statistical functions.
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
a.mean()  # 6.766666666666667

# Using a loop to generate summary statisics.

[
    func.__name__ + "=" + str(func())
    for func in (a.min, a.max, a.sum, a.prod, a.std, a.var)
]

#  ['min=-2.5',
#   'max=12.0',
#   'sum=40.6',
#   'prod=-71610.0',
#   'std=5.084835843520964',
#   'var=25.855555555555554']

# Using axis with operation along an axis.
c = np.arange(24).reshape(2, 3, 4)

#  array([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],
#
#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])

c.sum(axis=0)  # summing columns across matrices

#  array([[12, 14, 16, 18],
#         [20, 22, 24, 26],
#         [28, 30, 32, 34]])

c.sum(axis=1)  # summing rows across matrices

#  array([[12, 15, 18, 21],
#         [48, 51, 54, 57]])

c.sum(
    axis=(0, 2)
)  # 0+1+2+3 + 12+13+14+15, 4+5+6+7 + 16+17+18+19, 8+9+10+11 + 20+21+22+23
# array([ 60,  92, 124])

# Universal functions or ufunc, vectorized wrappers of simple functions.
s = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
np.square(s)  # array([[  6.25,   9.61,  49.  ], [100.  , 121.  , 144.  ]])

[{
    func.__name__: func(s)
} for func in (np.abs, np.sqrt, np.exp, np.log, np.sign, np.ceil, np.modf,
               np.isnan, np.cos)]

# Binary ufuncs apply elementwise on two ndarrays.
# Including broadcasting where needed.
a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])

np.add(a, b)  # array([ 3,  6,  2, 11])
np.greater(a, b)  # array([False, False,  True, False])
np.maximum(a, b)  # array([2, 8, 3, 7])
np.copysign(a, b)  # array([ 1.,  2., -3.,  4.])
