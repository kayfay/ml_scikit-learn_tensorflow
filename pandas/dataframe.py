import pandas as pd
import numpy as np

# Dataframes represent spreadsheets with cells, rows indexes,
# and columns labels. Define conditional expressions to compute columns
# Create pivot-tables, group rows, draw graphs, and others.
# Dataframes can be represented as dictionaries of series.
people_dict = {
    "weight":
    pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear":
    pd.Series(
        [1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children":
    pd.Series([0, 3], index=["charles", "bob"]),
    "hobby":
    pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}

people = pd.DataFrame(people_dict)
people

#           brithyear  children    hobby  weight
#  alice         1985       NaN   Biking      68  # Index alignment
#  bob           1984       3.0  Dancing      83  # Year data dropped
#  charles       1992       0.0      NaN     112  # Missing NaN

people["birthyear"]  # Display by column.

#  alice      1985
#  bob        1984
#  charles    1992
#  Name: birthyear, dtype: int64

people[["birthyear", "hobby"]]  # Display by multiple columns.

#           birthyear    hobby
#  alice         1985   Biking
#  bob           1984  Dancing
#  charles       1992      NaN

# Passing a list of columns and/or row labels/indexes, the constructor
# will create a DataFrame to those orders.
d = pd.DataFrame(
    people_dict,
    columns=["birthyear", "weight", "height"],
    index=["bob", "alice", "eugene"])

#          birthyear  weight height
#  bob        1984.0    83.0    NaN
#  alice      1985.0    68.0    NaN
#  eugene        NaN     NaN    NaN

# Creating a values to the constructor as an ndarray, list, or nested lists.

values = [[1985, np.nan, "Biking", 68], [1984, 3, "Dancing", 83],
          [1984, 0, np.nan, 112]]

d1 = pd.DataFrame(
    values,
    columns=["birthyear", "children", "hobby", "weight"],
    index=["alice", "bob", "charles"])

#           birthyear  children    hobby  weight
#  alice         1985       NaN   Biking      68
#  bob           1984       3.0  Dancing      83
#  charles       1984       0.0      NaN     112

# Masked arrays ignore NaN values.
masked_array = np.ma.asarray(values, dtype=np.object)
masked_array[(0, 2), (1, 2)] = np.ma.masked  # Mask coords

d2 = pd.DataFrame(
    masked_array,
    columns=["birthyear", "children", "hobby", "weight"],
    index=["alice", "bob", "charles"])

#          birthyear children    hobby weight
#  alice        1985      NaN   Biking     68
#  bob          1984        3  Dancing     83
#  charles      1984        0      NaN    112

# DataFrames instead of ndarray of values.

d3 = pd.DataFrame(d2, columns=["hobby", "children"], index=["alice", "bob"])

#           hobby children
#  alice   Biking      NaN
#  bob    Dancing        3

# DataFrames can also be created directly with dictionaries, lists, or nested versions.
people = pd.DataFrame({
    "birthyear": {
        "alice": 1985,
        "bob": 1984,
        "charles": 1992
    },
    "hobby": {
        "alice": "Biking",
        "bob": "Dancing"
    },
    "weight": {
        "alice": 68,
        "bob": 83,
        "charles": 112
    },
    "children": {
        "bob": 3,
        "charles": 0
    }
})

#           birthyear  children    hobby  weight
#  alice         1985       NaN   Biking      68
#  bob           1984       3.0  Dancing      83
#  charles       1992       0.0      NaN     112
