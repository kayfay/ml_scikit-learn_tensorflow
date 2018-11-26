import numpy as np
import pandas as pd

# CSV, Excel, JSON, HTML, HDF5, SQL
my_df = pd.DataFrame(
    [["Biking", 68.5, 1985, np.nan], ["Dancing", 83.1, 1984, 3]],
    columns=["hobby", "weight", "birthyear", "children"],
    index=["alice", "bob"])

my_df

#           hobby  weight  birthyear  children
#  alice   Biking    68.5       1985       NaN
#  bob    Dancing    83.1       1984       3.0

my_df.to_csv("my_df.csv")
my_df.to_html("my_df.html")
my_df.to_json("my_df.json")

for filename in ("my_df.csv", "my_df.html", "my_df.json"):
    print("#", filename)
    with open(filename, "rt") as f:
        print(f.read())
        print()

# my_df.csv
#  ,hobby,weight,birthyear,children # 1st col: Index.
#  alice,Biking,68.5,1985,
#  bob,Dancing,83.1,1984,3.0

# my_df.html
#  <table border="1" class="dataframe">
#    <thead>                            <!-- 1st col: Index -->
#      <tr style="text-align: right;">
#        <th></th>
#        <th>hobby</th>
#        <th>weight</th>
#        <th>birthyear</th>
#        <th>children</th>
#      </tr>
#    </thead>                          <!-- End col: Index  -->
#    <tbody>
#      <tr>
#        <th>alice</th>
#        <td>Biking</td>
#        <td>68.5</td>
#        <td>1985</td>
#        <td>NaN</td>
#      </tr>
#      <tr>
#        <th>bob</th>
#        <td>Dancing</td>
#        <td>83.1</td>
#        <td>1984</td>
#        <td>3.0</td>
#      </tr>
#    </tbody>
#  </table>

# my_df.json
#  {
#      "hobby": {
#          "alice": "Biking",
#          "bob": "Dancing"
#      },
#      "weight": {
#          "alice": 68.5,
#          "bob": 83.1
#      },
#      "birthyear": {
#          "alice": 1985,
#          "bob": 1984
#      },
#      "children": {
#          "alice": null,
#          "bob": 3.0
#      }
#  }

# Import module for Excel support.
try:
    my_df.to_excel("my_df.xlsx", sheet_name="People")
except ImportError as e:
    print(e)

# No module named 'openpyxl'
