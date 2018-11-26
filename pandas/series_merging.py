import numpy as np
import pandas as pd

weights = {"alice": 68, "bob": 83, "colin": 86, "darwin": 68}
s1 = pd.Series(weights)

#  alice     68
#  bob       83
#  colin     86
#  darwin    68
#  dtype: int64

s1.keys()

#  Index(['alice', 'bob', 'colin', 'darwin'], dtype='object')

s2 = pd.Series(weights, index=["colin", "alice"])  # Selecting elements.

#  colin    86
#  alice    68
#  dtype: int64

s2.keys()

#  Index(['colin', 'alice'], dtype='object')

# Initialize a series with a scalar value.
s3 = pd.Series(42, index=["joe", "charles", "alice"])

# Joining two series uses automatic alignment using matching index labels.
s2 + s3

#  alice      110.0
#  charles      NaN
#  colin        NaN
#  joe          NaN
#  dtype: float64

# Autoalignment requires labels, row labels will generate and join with new series.
s4 = pd.Series([1000, 1000, 1000, 1000])

# s2 =  [86 68]
# s4 =  [1000 1000 1000 1000]

s2 + s4

#  colin   NaN
#  alice   NaN
#  0       NaN
#  1       NaN
#  2       NaN
#  3       NaN
#  dtype: float64
