import pandas as pd

# Example of joins
# inner joins, left/right outer joins and full joins.

city_loc = pd.DataFrame(
    [["CA", "San Francisco", 37.781334, -122.416728], [
        "NY", "New York", 40.705649, -74.008344
    ], ["FL", "Miami", 25.791100, -80.320733], [
        "OH", "Cleveland", 41.473508, -81.739791
    ], ["UT", "Salt Lake City", 40.755851, -111.896657]],
    columns=["state", "city", "lat", "lng"])
city_loc

#    state            city        lat         lng
#  0    CA   San Francisco  37.781334 -122.416728
#  1    NY        New York  40.705649  -74.008344
#  2    FL           Miami  25.791100  -80.320733
#  3    OH       Cleveland  41.473508  -81.739791
#  4    UT  Salt Lake City  40.755851 -111.896657

city_pop = pd.DataFrame(
    [[808976, "San Francisco", "California"], [
        8363710, "New York", "New-York"
    ], [413201, "Miami", "Florida"], [2242193, "Houston", "Texas"]],
    index=[3, 4, 5, 6],
    columns=["population", "city", "state"])
city_pop

#     population           city       state
#  3      808976  San Francisco  California
#  4     8363710       New York    New-York
#  5      413201          Miami     Florida
#  6     2242193        Houston       Texas

# Merges INNER join drops cities that dont have existing values in both DataFrame.

pd.merge(left=city_loc, right=city_pop, on="city")

#    state_x           city        lat         lng  population     state_y
#  0      CA  San Francisco  37.781334 -122.416728      808976  California
#  1      NY       New York  40.705649  -74.008344     8363710    New-York
#  2      FL          Miami  25.791100  -80.320733      413201     Florida

# Merges FULL OUTER JOIN, no cities get dropped and NaN values are added.
all_cities = pd.merge(left=city_loc, right=city_pop, on="city", how="outer")
all_cities

#    state_x            city        lat         lng  population     state_y
#  0      CA   San Francisco  37.781334 -122.416728    808976.0  California
#  1      NY        New York  40.705649  -74.008344   8363710.0    New-York
#  2      FL           Miami  25.791100  -80.320733    413201.0     Florida
#  3      OH       Cleveland  41.473508  -81.739791         NaN         NaN
#  4      UT  Salt Lake City  40.755851 -111.896657         NaN         NaN
#  5     NaN         Houston        NaN         NaN   2242193.0       Texas

# IF the key to join is on both DataFrame indexes you must use
# left_index=True and/or right_index=True
# if the column names differ left_on and right_on

city_pop2 = city_pop.copy()
city_pop2.columns = ["population", "name", "state"]
pd.merge(left=city_loc, right=city_pop2, left_on="city", right_on="name")

#    state_x           city        lat         lng  population           name     state_y
#  0      CA  San Francisco  37.781334 -122.416728      808976  San Francisco  California
#  1      NY       New York  40.705649  -74.008344     8363710       New York    New-York
#  2      FL          Miami  25.791100  -80.320733      413201          Miami     Florida
