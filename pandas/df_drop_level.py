import numpy as np
import pandas as pd

# If all rows/columns are tuples of same size 
# they are undersood as multi-indexing.
d = pd.DataFrame({
    ("public", "birthyear"): {
        ("Paris", "alice"): 1985,
        ("Paris", "bob"): 1984,
        ("London", "charles"): 1992
    },
    ("public", "hobby"): {
        ("Paris", "alice"): "Biking",
        ("Paris", "bob"): "Dancing"
    },
    ("private", "weight"): {
        ("Paris", "alice"): 68,
        ("Paris", "bob"): 83,
        ("London", "charles"): 112
    },
    ("private", "children"): {
        ("Paris", "alice"): np.nan,
        ("Paris", "bob"): 3,
        ("London", "charles"): 0
    }
})

#                  private           public
#                 children weight birthyear    hobby
#  London charles      0.0    112      1992      NaN
#  Paris  alice        NaN     68      1985   Biking
#         bob          3.0     83      1984  Dancing

# Column levels; London and Paris and Index levels; private and public
# represent two levels, columns or indices can be dropped.

d.columns = d.columns.droplevel(level=0)

#                  children  weight  birthyear    hobby
#  London charles       0.0     112       1992      NaN
#  Paris  alice         NaN      68       1985   Biking
#         bob           3.0      83       1984  Dancing
