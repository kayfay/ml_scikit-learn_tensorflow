import numpy as np
import pandas as pd

much_data = np.fromfunction(lambda x, y: (x + y * y) % 17 * 11, (10000, 26))
large_df = pd.DataFrame(much_data, columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
large_df[large_df % 16 == 0] = np.nan
large_df.insert(3, "some_text", "Blabla")

# Show the head of the file.
large_df.head()

#        A     B     C some_text      D     E      F     G  ...       S     T      U     V      W     X      Y      Z
#  0   NaN  11.0  44.0    Blabla   99.0   NaN   88.0  22.0  ...    11.0  44.0   99.0   NaN   88.0  22.0  165.0  143.0
#  1  11.0  22.0  55.0    Blabla  110.0   NaN   99.0  33.0  ...    22.0  55.0  110.0   NaN   99.0  33.0    NaN  154.0
#  2  22.0  33.0  66.0    Blabla  121.0  11.0  110.0  44.0  ...    33.0  66.0  121.0  11.0  110.0  44.0    NaN  165.0
#  3  33.0  44.0  77.0    Blabla  132.0  22.0  121.0  55.0  ...    44.0  77.0  132.0  22.0  121.0  55.0   11.0    NaN
#  4  44.0  55.0  88.0    Blabla  143.0  33.0  132.0  66.0  ...    55.0  88.0  143.0  33.0  132.0  66.0   22.0    NaN
#
#  [5 rows x 27 columns]

# Show the tail of the file.
large_df.tail(n=2)

#           A     B     C some_text      D     E      F     G  ...       S     T      U     V      W     X     Y      Z
#  9998  22.0  33.0  66.0    Blabla  121.0  11.0  110.0  44.0  ...    33.0  66.0  121.0  11.0  110.0  44.0   NaN  165.0
#  9999  33.0  44.0  77.0    Blabla  132.0  22.0  121.0  55.0  ...    44.0  77.0  132.0  22.0  121.0  55.0  11.0    NaN
#
#  [2 rows x 27 columns]

# Print out a summary of each columns contents
large_df.info()

#  <class 'pandas.core.frame.DataFrame'>
#  RangeIndex: 10000 entries, 0 to 9999
#  Data columns (total 27 columns):
#  A            8823 non-null float64
#  B            8824 non-null float64
#  C            8824 non-null float64
#  some_text    10000 non-null object
#  D            8824 non-null float64
#  E            8822 non-null float64
#  F            8824 non-null float64
#  G            8824 non-null float64
#  H            8822 non-null float64
#  I            8823 non-null float64
#  J            8823 non-null float64
#  K            8822 non-null float64
#  L            8824 non-null float64
#  M            8824 non-null float64
#  N            8822 non-null float64
#  O            8824 non-null float64
#  P            8824 non-null float64
#  Q            8824 non-null float64
#  R            8823 non-null float64
#  S            8824 non-null float64
#  T            8824 non-null float64
#  U            8824 non-null float64
#  V            8822 non-null float64
#  W            8824 non-null float64
#  X            8824 non-null float64
#  Y            8822 non-null float64
#  Z            8823 non-null float64
#  dtypes: float64(26), object(1)
#  memory usage: 2.1+ MB

# Find main aggregated values of each column;
# count, mean, std, min, quantiles, max, (non-null).
large_df.describe()

#                   A            B            C     ...                 X            Y            Z
#  count  8823.000000  8824.000000  8824.000000     ...       8824.000000  8822.000000  8823.000000
#  mean     87.977559    87.972575    87.987534     ...         87.977561    88.000000    88.022441
#  std      47.535911    47.535523    47.521679     ...         47.529755    47.536879    47.535911
#  min      11.000000    11.000000    11.000000     ...         11.000000    11.000000    11.000000
#  25%      44.000000    44.000000    44.000000     ...         44.000000    44.000000    44.000000
#  50%      88.000000    88.000000    88.000000     ...         88.000000    88.000000    88.000000
#  75%     132.000000   132.000000   132.000000     ...        132.000000   132.000000   132.000000
#  max     165.000000   165.000000   165.000000     ...        165.000000   165.000000   165.000000
#
#  [8 rows x 26 columns]
