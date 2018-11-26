import numpy as np
import pandas as pd

# If all rows/columns are tuples of same size they are undersood as multi-indexing.
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

# Push the lowest column level after the lowest index
d.stack()

#                            private   public
#  London charles birthyear      NaN     1992
#                 children       0.0      NaN
#                 weight       112.0      NaN
#  Paris  alice   birthyear      NaN     1985
#                 hobby          NaN   Biking
#                 weight        68.0      NaN
#         bob     birthyear      NaN     1984
#                 children       3.0      NaN
#                 hobby          NaN  Dancing
#                 weight        83.0      NaN

# Put the lowest index after the lowest column
d.unstack()

#          private                                      public
#         children              weight               birthyear                   hobby
#            alice  bob charles  alice   bob charles     alice     bob charles   alice      bob charles
#  London      NaN  NaN     0.0    NaN   NaN   112.0       NaN     NaN  1992.0     NaN      NaN     NaN
#  Paris       NaN  3.0     NaN   68.0  83.0     NaN    1985.0  1984.0     NaN  Biking  Dancing     NaN

# Unstacking down and down createst a Series object
d.unstack().unstack()

#  private  children   alice    London        NaN
#                               Paris         NaN
#                      bob      London        NaN
#                               Paris           3
#                      charles  London          0
#                               Paris         NaN
#           weight     alice    London        NaN
#                               Paris          68
#                      bob      London        NaN
#                               Paris          83
#                      charles  London        112
#                               Paris         NaN
#  public   birthyear  alice    London        NaN
#                               Paris        1985
#                      bob      London        NaN
#                               Paris        1984
#                      charles  London       1992
#                               Paris         NaN
#           hobby      alice    London        NaN
#                               Paris      Biking
#                      bob      London        NaN
#                               Paris     Dancing
#                      charles  London        NaN
#                               Paris         NaN
#  dtype:object

# Or you can use level and pass a tuple of
# indexes to unstack as parameters.
d.unstack(level=(0, 1))

#  private  children   London  alice          NaN
#                              bob            NaN
#                              charles          0
#                      Paris   alice          NaN
#                              bob              3
#                              charles        NaN
#           weight     London  alice          NaN
#                              bob            NaN
#                              charles        112
#                      Paris   alice           68
#                              bob             83
#                              charles        NaN
#  public   birthyear  London  alice          NaN
#                              bob            NaN
#                              charles       1992
#                      Paris   alice         1985
#                              bob           1984
#                              charles        NaN
#           hobby      London  alice          NaN
#                              bob            NaN
#                              charles        NaN
#                      Paris   alice       Biking
#                              bob        Dancing
#                              charles        NaN
#  dtype: object
