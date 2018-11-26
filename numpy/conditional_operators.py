import numpy as np

# Conditional operators are elementwise
m = np.array([20, -5, 30, 40])
m < [15, 16, 35, 36]
# array([False,  True,  True, False])

# Broadcasting can also be used
m < 25
# array([ True,  True, False, False])

# Most useful in conjunction with boolean indexing
m[m < 25]
# array([20, -5])
