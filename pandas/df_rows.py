import pandas as pd
import numpy as np

people = pd.DataFrame({
    "birthyear": {
        "alice": 1985,
        "bob": 1984,
        "charles": 1992
    },
    "hobby": {
        "alice": "Biking",
        "bob": "Dancnig"
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

people

#           birthyear  children    hobby  weight
#  alice         1985       NaN   Biking      68
#  bob           1984       3.0  Dancnig      83
#  charles       1992       0.0      NaN     112

# Accessing rows.

people.loc["charles"]

#  birthyear    1992
#  children        0
#  hobby         NaN
#  weight        112
#  Name: charles, dtype: object

people.iloc[2]

#  birthyear    1992
#  children        0
#  hobby         NaN
#  weight        112
#  Name: charles, dtype: object

people.iloc[1:3]

#            birthyear  children    hobby  weight
#  bob           1984       3.0  Dancnig      83
#  charles       1992       0.0      NaN     112

people[np.array([True, False, True])]

#            hobby  height  weight  age  over 30  pets
#  alice    Biking     172      68   33     True   NaN
#  charles     NaN     185     112   26    False   5.0

people[people["birthyear"] < 1990]

#         birthyear  children    hobby  weight
#  alice       1985       NaN   Biking      68
#  bob         1984       3.0  Dancnig      83

# Adding and removing rows.

people["age"] = 2018 - people["birthyear"]
people["over 30"] = people["age"] > 30
birthyears = people.pop("birthyear")
del people["children"]

people

#             hobby  weight  age  over 30
#  alice     Biking      68   33     True
#  bob      Dancnig      83   34     True
#  charles      NaN     112   26    False

birthyears

#  alice      1985
#  bob        1984
#  charles    1992
#  Name: birthyear, dtype: int64

# Adding columns, must have equal rows, missing
# rows are filled with NaN and extras are ignored.

people["pets"] = pd.Series({"bob": 0, "charles": 5, "eugene": 1})
people

#             hobby  weight  age  over 30  pets
#  alice     Biking      68   33     True   NaN
#  bob      Dancnig      83   34     True   0.0
#  charles      NaN     112   26    False   5.0

# New columns added at the end, insert passing the first argument
# as index for location.
people.insert(1, "height", [172, 181, 185])
people

#             hobby  height  weight  age  over 30  pets
#  alice     Biking     172      68   33     True   NaN
#  bob      Dancnig     181      83   34     True   0.0
#  charles      NaN     185     112   26    False   5.0

# Assigning new columns returning a new DataFrame object.
people.assign(
    body_mass_index=people["weight"] / (people["height"] / 100)**2,
    has_pets=people["pets"] > 0)

#           birthyear  height  children    hobby  weight  pets  body_mass_index  has_pets
#  alice         1985     172       NaN   Biking      68   NaN        22.985398     False
#  bob           1984     181       3.0  Dancnig      83   0.0        25.335002     False
#  charles       1992     185       0.0      NaN     112   5.0        32.724617      True

# assignment does not access existing columns
try:
    people.assign(
        body_mass_index=people["weight"] / (people["height"] / 100)**2,
        overweight=people["body_mass_index"] > 25)
except KeyError as e:
    print("Key error:", e)

#  'body_mass_index'

# Splitting it into a variable and then a second assign works.
d1 = people.assign(body_mass_index=people["weight"] /
                   (people["height"] / 100)**2)
d1.assign(overweight=d1["body_mass_index"] > 25)

#           birthyear  height  children    hobby  weight  pets  body_mass_index  overweight
#  alice         1985     172       NaN   Biking      68   NaN        22.985398       False
#  bob           1984     181       3.0  Dancnig      83   0.0        25.335002        True
#  charles       1992     185       0.0      NaN     112   5.0        32.724617        True

# Or pass a function (lambda) to do assignment.
(people.assign(body_mass_index=lambda df: df["weight"] / (df["height"])**2)
 .assign(overweight=lambda df: df["body_mass_index"] > 25))
