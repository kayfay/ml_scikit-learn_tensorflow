import numpy as np
import pandas as pd

# Create DataFrame Components
values = np.array([[8, 8, 9, np.nan], [10, 9, 10, np.nan], [4, 11, 5, np.nan],
                   [np.nan, np.nan, np.nan, np.nan], [9, 10, 11, np.nan]])
col = ["sep", "oct", "nov", "dec"]
row = ["alice", "bob", "charles", "colin", "darwin"]

final_grades = pd.DataFrame(values, index=row, columns=col)
final_grades["hobby"] = ["Biking", "Dancing", np.nan, "Dancing", "Biking"]
final_grades

grouped_grades = final_grades.groupby("hobby")
grouped_grades

# <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x7f85a6f55828>

# Average grade by hobby.
grouped_grades.mean()
