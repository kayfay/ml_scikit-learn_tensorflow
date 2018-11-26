import numpy as np
import pandas as pd

# Create DataFrame Components.
values = np.array([[8, 8, 9], [10, 9, 10], [4, 11, 5], [9, 10, 11]])
col = ["sep", "oct", "nov"]
row = ["alice", "bob", "colin", "darwin"]

# Create a dataframe with more grades added and flattened.
final_grades = pd.DataFrame(values, index=row, columns=col)
more_grades = final_grades.stack().reset_index()
more_grades.columns = ["name", "month", "grade"]
more_grades["bonus"] = [np.nan, np.nan, np.nan, 0, np.nan, 2, 3, 3, 0, 0, 1, 0]
more_grades

# Compute mean.
pd.pivot_table(more_grades, index="name")

#             bonus      grade
#  name
#  alice        NaN   8.333333
#  bob     1.000000   9.666667
#  colin   2.000000   6.666667
#  darwin  0.333333  10.000000

# Specify a list of columns and a function to aggergate by.
pd.pivot_table(
    more_grades, index="name", values=["grade", "bonus"], aggfunc=np.max)

#          bonus  grade
#  name
#  alice     NaN      9
#  bob       2.0     10
#  colin     3.0     11
#  darwin    1.0     11

# Creating multilevel indices.
pd.pivot_table(more_grades, index=("name", "month"), margins=True)

#                bonus  grade
#
#  name   month
#  alice  nov      NaN      9
#         oct      NaN      8
#         sep      NaN      8
#  bob    nov    2.000     10
#         oct      NaN      9
#         sep    0.000     10
#  colin  nov    0.000      5
#         oct    3.000     11
#         sep    3.000      4
#  darwin nov    0.000     11
#         oct    1.000     10
#         sep    0.000      9
#  All           1.125      8
