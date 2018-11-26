import pandas as pd

# Series can also use names
s = pd.Series([83, 68], index=["bob", "alice"], name="weights")

#  bob      83
#  alice    68
#  Name: weights, dtype: int64
