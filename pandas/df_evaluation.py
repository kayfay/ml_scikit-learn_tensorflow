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
        "alice":True, 
        "bob":True, 
        "charles":False
    },
    "body_mass_index": {
        "alice": 22.985,
        "bob": 25.335,
        "charles": 32.724
    },
    "pets": {
        "bob": 0, 
        "charles": 5
    }
})

people

#           age  body_mass_index  height    hobby  over 30  pets  weight
#  alice     33           22.985     172   Biking     True   NaN      68
#  bob       34           25.335     181  Dancnig     True   0.0      83
#  charles   26           32.724     185      NaN    False   5.0     112

# Expression validation
people.eval("weight / (height/100) ** 2 > 25")

#  alice      False
#  bob         True
#  charles     True
#  dtype: bool

# Assignment expressions directly to a DataFrame.
people.eval("body_mass_index = weight / (height/100) ** 2", inplace = True)
people

#           age  body_mass_index  height    hobby  over 30  pets  weight
#  alice     33        22.985398     172   Biking     True   NaN      68
#  bob       34        25.335002     181  Dancnig     True   0.0      83
#  charles   26        32.724617     185      NaN    False   5.0     112

overweight_threshold = 30
people.eval("overweight = body_mass_index > @overweight_threshold", inplace=True)
people

#           age  body_mass_index  height    hobby  over 30  pets  weight  overweight
#  alice     33        22.985398     172   Biking     True   NaN      68       False
#  bob       34        25.335002     181  Dancnig     True   0.0      83       False
#  charles   26        32.724617     185      NaN    False   5.0     112        True
