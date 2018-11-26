import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

people.sort_index(axis=1, inplace=True)
people.sort_values(by="age", inplace=True)
people

# Plotting a DataFrame.
people.plot(kind="line", x="body_mass_index", y=["height", "weight"])
plt.show()

# Scatter plot with a list of sizes.
people.plot(kind = "scatter", x = "height", y = "weight", s=[40, 120, 200])
plt.show()
