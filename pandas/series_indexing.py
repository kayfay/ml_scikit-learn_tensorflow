import pandas as pd
import numpy as np

# Series is an object a 1D array with indices
# Similar to a column with row labels.
s = pd.Series([2, -1, 3, 5])

#  0    2
#  1   -1
#  2    3
#  3    5
#  dtype: int64


# Act like np.ndarray.
np.exp(s)

#  0      7.389056
#  1      0.367879
#  2     20.085537
#  3    148.413159
#  dtype: float64



# Arithmetic operates elementwise like an ndarray.
s + [1000, 2000, 3000, 4000]

#  0    1002
#  1    1999
#  2    3003
#  3    4005
#  dtype: int64

# Conditional Operations
s < 0

#  0    False
#  1     True
#  2    False
#  3    False
#  dtype: bool

# Labels can be indexed, and manipulated.
s2 = pd.Series([68, 83, 112, 68], index=["alice", "bob", "charles", "darwin"])

#  alice       68
#  bob         83
#  charles    112
#  darwin      68
#  dtype: int64

# Indexing can be done by key or number, for readability use loc/iloc.
s2["bob"] # 83 
s2[1] # 83

# Prefered method for label location or index location.
s2.loc["bob"]
s2.iloc[1]

# Slice a series using index labels.
s2.iloc[1:3]

#  bob         83
#  charles    112
#  dtype: int64

# Note that indexes start with zero and slicing is index based.
rows = pd.Series([1000, 1001, 1002, 1003])

#  0    1000
#  1    1001
#  2    1002
#  3    1003
#  dtype: int64

# Slices retain indexes from a series, as being treated a row.
row_slice = rows[2:]

try:
    row_slice[0] # slice at 0
except KeyError as e:
    print("Key error:", e)

#  Key error: 0

# In a series index location is the prefered method. 
row_slice.iloc[0] # 1002
