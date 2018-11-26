import numpy as np
import pandas as pd

people = pd.DataFrame({
    "hobby": {
        "alice": "Biking",
        "bob": "Dancnig"
    },
    "height": {
        "alice": 172,
        "bob": 181,
        "charles": 185
    },
    "age": {
        "alice": 33,
        "bob": 34,
        "charles": 26
    },
    "weight": {
        "alice": 68,
        "bob": 83,
        "charles": 112
    },
    "over 30": {
        "alice": True,
        "bob": True,
        "charles": False
    },
    "body_mass_index": {
        "alice": 22.985398,
        "bob": 25.335002,
        "charles": 32.724617
    },
    "pets": {
        "bob": 0,
        "charles": 5
    },
    "overweight": {
        "alice": False,
        "bob": False,
        "charles": True
    }
})

# Querying a DataFrame based on a query expression.
people.query("age > 30 and pets == 0")

#        age  body_mass_index  height    hobby over 30 overweight  pets  weight
#  bob  34.0        25.335002   181.0  Dancnig    True        NaN   0.0    83.0

# Sorting a DataFrame sort rows by their index label,
# in ascending order, or reverse order.
people.sort_index(ascending=False)

#           age  body_mass_index  height    hobby  over 30  overweight  pets  weight
#  charles   26        32.724617     185      NaN    False        True   5.0     112
#  bob       34        25.335002     181  Dancnig     True       False   0.0      83
#  alice     33        22.985398     172   Biking     True       False   NaN      68

# Sort in place and by columns.
people.sort_index(axis=1, inplace=True)

# Sort by values instead of labels
people.sort_values(by="age", inplace=True)
