import numpy as np
import pandas as pd

# If all rows/columns are tuples of same size they are undersood as multi-indexing.
d = pd.DataFrame(
    {

        ("public", "birthyear"):
        {("Paris", "alice"): 1985, ("Paris", "bob"): 1984, ("London", "charles"): 1992},
        ("public", "hobby"):
        {("Paris", "alice"): "Biking", ("Paris", "bob"): "Dancing"},
        ("private", "weight"):
        {("Paris", "alice"): 68, ("Paris", "bob"): 83, ("London", "charles"): 112},
        ("private", "children"):
        {("Paris", "alice"): np.nan, ("Paris", "bob"): 3, ("London", "charles"): 0}
    }
)

d

#                  private           public
#                 children weight birthyear    hobby
#  London charles      0.0    112      1992      NaN
#  Paris  alice        NaN     68      1985   Biking
#         bob          3.0     83      1984  Dancing


d = d.T

#                     London   Paris
#                    charles   alice      bob
#  private children        0     NaN        3
#          weight        112      68       83
#  public  birthyear    1992    1985     1984
#          hobby         NaN  Biking  Dancing

