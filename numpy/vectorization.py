import numpy as np

# Vectorization mathmatics example *effeciency
# Same as manually calculating iteritavely
# Performing as numerical operations (calculations)

import math
data = np.empty((768, 1024))
for y in range(768):
    for x in range(1024):
        data[y, x] = math.sin(x * y / 40.5)

data = None

# Generate coordinate matricies
x_coords = np.arange(0, 1024)
y_coords = np.arange(0, 768)

X, Y = np.meshgrid(x_coords, y_coords)

#  X

#  array([[   0,    1,    2, ..., 1021, 1022, 1023],
#         [   0,    1,    2, ..., 1021, 1022, 1023],
#         [   0,    1,    2, ..., 1021, 1022, 1023],
#         ...,
#         [   0,    1,    2, ..., 1021, 1022, 1023],
#         [   0,    1,    2, ..., 1021, 1022, 1023],
#         [   0,    1,    2, ..., 1021, 1022, 1023]])

#  Y

#  array([[  0,   0,   0, ...,   0,   0,   0],
#         [  1,   1,   1, ...,   1,   1,   1],
#         [  2,   2,   2, ...,   2,   2,   2],
#         ...,
#         [765, 765, 765, ..., 765, 765, 765],
#         [766, 766, 766, ..., 766, 766, 766],
#         [767, 767, 767, ..., 767, 767, 767]])

data = np.sin(X * Y / 40.5)
