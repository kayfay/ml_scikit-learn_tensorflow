import numpy as np

# Create array of zeros.
np.zeros(5)

# Arrays have a type of ndarray
type(np.zeros(5))
# In [18]: Out[18]: numpy.ndarray

#  In [3]: Out[3]: array([0., 0., 0., 0., 0.])

# Create a 2D array (i.e., a matrix).
array_2D = np.zeros((3, 4))  # 3 by 4, tuple with rows, cols.

#  array([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])
#

# Each dimension is called an axis.
array_2D.shape  # (3, 4)
# Each axes makes up the number of ranks.
array_2D.ndim  # 2
# Axis 1 has a length of 3.
# Axis 2 has a length of 4.
array_2D.size  # 12

# Create a 3D array, N-dimensional array.
array_3D = np.zeros((2, 3, 4))  # Rank = 3

#  array([[[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],
#
#         [[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]]])

# The number of axes is called the rank.
array_3D.shape  # (3, 4, 2) Rank 2, (array indexes start at 0)

# NumPy functions for array creations.
np.ones((3, 4))

#  array([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]])

# Initialize a shape given a value.
np.full((3, 4), np.pi)

#  array([[3.14159265, 3.14159265, 3.14159265, 3.14159265],
#         [3.14159265, 3.14159265, 3.14159265, 3.14159265],
#         [3.14159265, 3.14159265, 3.14159265, 3.14159265]])

# Allocates space directly from memory.
np.empty((2, 3))

#  array([[4.64317662e-310, 4.64317576e-310, 4.64317575e-310],
#         [6.91849759e-310, 0.00000000e+000, 0.00000000e+000]])

# Initialize given shape and values.
np.array([[1, 2, 3, 4], [10, 20, 30, 40]])

#  array([[ 1,  2,  3,  4],
#        [10, 20, 30, 40]])

# Array range from x, up to y.
np.arange(1, 5)  # array([1, 2, 3, 4])
np.arange(1., 5.)  # array([1., 2., 3., 4.])
np.arange(1, 5, 0.5)  # array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])

# Linspace is preferable compared arange.
# In example trying to create equally spaced ranges
# Doesn't work because of floating point errors

np.arange(0, 5/3, 1/3)  
#  array([0.        , 0.33333333, 0.66666667, 1.        , 1.33333333,
#         1.66666667])

np.arange(0, 5/3, 0.33333334)
# array([0.        , 0.33333334, 0.66666668, 1.00000002, 1.33333336])

# Create specified equally spaced distances between two values.
np.linspace(0, 5/3, 6)
#  array([0.        , 0.33333333, 0.66666667, 1.        , 1.33333333,
#         1.66666667])

# Random numbers between 0 and 1, uniform distribution.
np.random.rand(3, 4) 
#  array([[8.90440793e-01, 1.55668692e-01, 7.82163476e-04, 1.05887455e-01],
#         [7.12315529e-01, 8.89731282e-01, 3.03282832e-02, 8.24481755e-02],
#         [7.19961034e-01, 8.15866067e-01, 8.87782859e-02, 6.13324802e-01]])

# Random numbers mean 0 and variance 1.
# Univariate normal distribution, (Gaussian distribution).
np.random.randn(3, 4)
#  array([[ 0.60911469, -0.78076729,  1.5894877 ,  0.1209884 ],
#         [-0.99705368, -0.91459414, -0.42247904, -0.4067462 ],
#         [ 1.86495808,  0.12347278, -0.18124363, -0.51739468]])

# Initialize an array from a function.
np.fromfunction(lambda z, y, x: x * y + z, (3, 2, 10))

#  array([[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#          [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]],
#
#         [[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
#          [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]],
#
#         [[ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
#          [ 2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.]]])

# Creates 3 ndarrays one per dimension, with shape 2, 10
# Each array has values equal to the coordinate along the specific axis.

# Array data type dtype.
array_2D.dtype # dtype('float64')

#  array([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])


# dtypes include int8, int16, int32, int64, uint8|16|32|64,
# float16|32|64 complex64|128
np.arange(1, 5, dtype=np.complex64)

# array([1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j], dtype=complex64)

# Size in bytes of each item
np.arange(1, 5, dtype=np.complex64).itemsize # 8

# Show array's stored memory byte buffer
np.array([[1, 2], [1000, 2000]], dtype=np.int32).data
# <memory at 0x7f5bbb016dc8>
