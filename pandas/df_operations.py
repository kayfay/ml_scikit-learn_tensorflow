import numpy as np
import pandas as pd

grades_array = np.array([[8, 8, 9], [10, 9, 9], [4, 8, 2], [9, 10, 10]])

grades = pd.DataFrame(
    grades_array,
    columns=["sep", "oct", "nov"],
    index=["alice", "bob", "charles", "darwin"])

grades

# Performing operations on arrays.
np.sqrt(grades)

#                sep       oct       nov
#  alice    2.828427  2.828427  3.000000
#  bob      3.162278  3.000000  3.000000
#  charles  2.000000  2.828427  1.414214
#  darwin   3.000000  3.162278  3.162278

# Performing broadcasting on arrays, elementwise
# operations on each value.

grades + 1

#           sep  oct  nov
#  alice      9    9   10
#  bob       11   10   10
#  charles    5    9    3
#  darwin    10   11   11

grades >= 5

#             sep   oct    nov
#  alice     True  True   True
#  bob       True  True   True
#  charles  False  True  False
#  darwin    True  True   True

# Aggeration options such as max, sum, mean.
grades.mean()

#  sep    7.75
#  oct    8.75
#  nov    7.50
#  dtype: float64

# Check values for truth.
(grades > 5).all()

#  sep    False
#  oct     True
#  nov    False
#  dtype: bool

# Axis parameters specifys an operation execution on a DataFrame
# axis=0 vertical execution, axis=1 horizontal execution
(grades > 5).all(axis=1)  # all students had grades greater than 5

#  alice       True
#  bob         True
#  charles    False
#  darwin      True
#  dtype: bool

# Any returns True if a value is True.
(grades == 10).any(axis=1)  # Find any higher grades than 10

#  alice      False
#  bob         True
#  charles    False
#  darwin      True
#  dtype: bool

# Binary operations broadcast to all rows, e.g., subtract mean.
grades - grades.mean()

#            sep   oct  nov
#  alice    0.25 -0.75  1.5
#  bob      2.25  0.25  1.5
#  charles -3.75 -0.75 -5.5
#  darwin   1.25  1.25  2.5

# Subtracting the global mean.

grades - grades.values.mean()

#           sep  oct  nov
#  alice    0.0  0.0  1.0
#  bob      2.0  1.0  1.0
#  charles -4.0  0.0 -6.0
#  darwin   1.0  2.0  2.0
