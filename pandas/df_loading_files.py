import numpy as np
import pandas as pd


my_df_loaded = pd.read_csv("my_df.csv", index_col=0)
my_df_loaded

us_cities = None
try:
    csv_url = "https://raw.githubusercontent.com/datasets/world-cities/master/data/world-cities.csv"
    us_cities = pd.read_csv(csv_url, index_col=0)
    us_cities = us_cities.head()
except IOError as e:
    print(e)
us_cities

#                                 country          subcountry  geonameid
#  name
#  les Escaldes                   Andorra  Escaldes-Engordany    3040051
#  Andorra la Vella               Andorra    Andorra la Vella    3041563
#  Umm al Qaywayn    United Arab Emirates      Umm al Qaywayn     290594
#  Ras al-Khaimah    United Arab Emirates     Raʼs al Khaymah     291074
#  Khawr Fakkān      United Arab Emirates        Ash Shāriqah     291696


