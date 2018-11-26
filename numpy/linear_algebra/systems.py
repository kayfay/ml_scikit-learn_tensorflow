import numpy as np
from numpy import linalg

# Solving linear scalar equations, e.g., 2x+6y=6, 5x+3y=-9
coeffs = np.array([[2, 6], [5, 3]])
depvars = np.array([6, -9])
solution = linalg.solve(coeffs, depvars)
solution  # x, y

#  array([-3.,  2.])

# Comparing x, y solution
coeffs.dot(solution), depvars

#  (array([ 6., -9.]), array([ 6, -9]))

# Test x, y solution
np.allclose(coeffs.dot(solution), depvars)

#  True
