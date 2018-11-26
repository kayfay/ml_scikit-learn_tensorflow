import numpy as np
import pandas as pd

grades_array = np.array([[8, 8, 9], [10, 9, 9], [4, 8, 2], [9, 10, 10]])

grades = pd.DataFrame(
    grades_array,
    columns=["sep", "oct", "nov"],
    index=["alice", "bob", "charles", "darwin"])

grades

#           sep  oct  nov
#  alice      8    8    9
#  bob       10    9    9
#  charles    4    8    2
#  darwin     9   10   10



# Adding bonus points to the grades.
bonus_array = np.array([[0, np.nan,2],[np.nan,1,0],[0,1,0],[3,3,0]])
bonus_points = pd.DataFrame(bonus_array, columns=["oct", "nov", "dec"],
                            index=["bob", "colin", "darwin", "charles"])
bonus_points

#           oct  nov  dec
#  bob      0.0  NaN  2.0
#  colin    NaN  1.0  0.0
#  darwin   0.0  1.0  0.0
#  charles  3.0  3.0  0.0

grades + bonus_points

#           dec   nov   oct  sep
#  alice    NaN   NaN   NaN  NaN
#  bob      NaN   NaN   9.0  NaN
#  charles  NaN   5.0  11.0  NaN
#  colin    NaN   NaN   NaN  NaN
#  darwin   NaN  11.0  10.0  NaN


