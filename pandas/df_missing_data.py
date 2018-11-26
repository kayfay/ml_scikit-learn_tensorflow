import numpy as np
import pandas as pd

grades_array = np.array([[8, 8, 9], [10, 9, 9], [4, 8, 2], [9, 10, 10]])

grades = pd.DataFrame(
    grades_array,
    columns=["sep", "oct", "nov"],
    index=["alice", "bob", "charles", "darwin"])

bonus_array = np.array([[0, np.nan, 2], [np.nan, 1, 0], [0, 1, 0], [3, 3, 0]])
bonus_points = pd.DataFrame(
    bonus_array,
    columns=["oct", "nov", "dec"],
    index=["bob", "colin", "darwin", "charles"])
grades + bonus_points

#           dec   nov   oct  sep
#  alice    NaN   NaN   NaN  NaN
#  bob      NaN   NaN   9.0  NaN
#  charles  NaN   5.0  11.0  NaN
#  colin    NaN   NaN   NaN  NaN
#  darwin   NaN  11.0  10.0  NaN

# Handling missing data, assigning zeros for NaN values.
(grades + bonus_points).fillna(0)

#           dec   nov   oct  sep
#  alice    0.0   0.0   0.0  0.0
#  bob      0.0   0.0   9.0  0.0
#  charles  0.0   5.0  11.0  0.0
#  colin    0.0   0.0   0.0  0.0
#  darwin   0.0  11.0  10.0  0.0

# Setting missing grades to zero is unfair for students
# missing grades and missing bonus points are should be
# na and zero.

fixed_bonus_points = bonus_points.fillna(0)
fixed_bonus_points.insert(0, "sep", 0)
fixed_bonus_points.loc["alice"] = 0
grades + fixed_bonus_points

#           dec   nov   oct   sep
#  alice    NaN   9.0   8.0   8.0
#  bob      NaN   9.0   9.0  10.0
#  charles  NaN   5.0  11.0   4.0
#  colin    NaN   NaN   NaN   NaN
#  darwin   NaN  11.0  10.0   9.0

bonus_points

#           oct  nov  dec
#  bob      0.0  NaN  2.0
#  colin    NaN  1.0  0.0
#  darwin   0.0  1.0  0.0
#  charles  3.0  3.0  0.0

# Interpolation takes the mean from left and right.
bonus_points.interpolate(axis=1)

#           oct  nov  dec
#  bob      0.0  1.0  2.0
#  colin    NaN  1.0  0.0
#  darwin   0.0  1.0  0.0
#  charles  3.0  3.0  0.0

# Adding in sepetember column to fill Nan.
better_bonus_points = bonus_points.copy()
better_bonus_points.insert(0, "sep", 0)
better_bonus_points.loc["alice"] = 0
better_bonus_points = better_bonus_points.interpolate(axis=1)
better_bonus_points

#           sep  oct  nov  dec
#  bob      0.0  0.0  1.0  2.0
#  colin    0.0  0.5  1.0  0.0
#  darwin   0.0  0.0  1.0  0.0
#  charles  0.0  3.0  3.0  0.0
#  alice    0.0  0.0  0.0  0.0

grades["dec"] = np.nan
final_grades = grades + better_bonus_points
final_grades

#            sep   oct   nov  dec
#  alice     8.0   8.0   9.0  NaN
#  bob      10.0   9.0  10.0  NaN
#  charles   4.0  11.0   5.0  NaN
#  colin     NaN   NaN   NaN  NaN
#  darwin    9.0  10.0  11.0  NaN

final_grades_clean = final_grades.dropna(how="all")
final_grades_clean

#            sep   oct   nov  dec
#  alice     8.0   8.0   9.0  NaN
#  bob      10.0   9.0  10.0  NaN
#  charles   4.0  11.0   5.0  NaN
#  darwin    9.0  10.0  11.0  NaN

final_grades_clean = final_grades_clean.dropna(axis=1, how="all")
final_grades_clean

#            sep   oct   nov
#  alice     8.0   8.0   9.0
#  bob      10.0   9.0  10.0
#  charles   4.0  11.0   5.0
#  darwin    9.0  10.0  11.0
