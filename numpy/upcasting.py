import numpy as np

# Data types, dtype, is upcasted to a type capable of handling values
k1 = np.arange(0, 5, dtype=np.uint8)
print(k1.dtype, k1)
# uint8 [0 1 2 3 4]

k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print(k2.dtype, k2)
# int16 [ 5  7  9 11 13]

k3 = k1 + 1.5
print(k3.dtype, k3)
# float64 [1.5 2.5 3.5 4.5 5.5]
