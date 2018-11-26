import pandas as pd

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

# Examples of concatenation of two DataFrame objects.
result_concat = pd.concat([city_loc, city_pop])
result_concat

#               city        lat         lng  population       state
#  0   San Francisco  37.781334 -122.416728         NaN          CA
#  1        New York  40.705649  -74.008344         NaN          NY
#  2           Miami  25.791100  -80.320733         NaN          FL
#  3       Cleveland  41.473508  -81.739791         NaN          OH
#  4  Salt Lake City  40.755851 -111.896657         NaN          UT
#  3   San Francisco        NaN         NaN    808976.0  California
#  4        New York        NaN         NaN   8363710.0    New-York
#  5           Miami        NaN         NaN    413201.0     Florida
#  6         Houston        NaN         NaN   2242193.0       Texas

# Concatenation aligned two column threes.
result_concat.loc[3]

#              city        lat        lng  population       state
#  3      Cleveland  41.473508 -81.739791         NaN          OH
#  3  San Francisco        NaN        NaN    808976.0  California

# Setting ignore index option in concatenation.
pd.concat([city_loc, city_pop], ignore_index=True)

#               city        lat         lng  population       state
#  0   San Francisco  37.781334 -122.416728         NaN          CA
#  1        New York  40.705649  -74.008344         NaN          NY
#  2           Miami  25.791100  -80.320733         NaN          FL
#  3       Cleveland  41.473508  -81.739791         NaN          OH
#  4  Salt Lake City  40.755851 -111.896657         NaN          UT
#  5   San Francisco        NaN         NaN    808976.0  California
#  6        New York        NaN         NaN   8363710.0    New-York
#  7           Miami        NaN         NaN    413201.0     Florida
#  8         Houston        NaN         NaN   2242193.0       Texas

# Setting inner results in an NaN values being removed by
# only including columns and rows that exist in both dataframes.

pd.concat([city_loc, city_pop], join="inner")

#          state            city
#  0          CA   San Francisco
#  1          NY        New York
#  2          FL           Miami
#  3          OH       Cleveland
#  4          UT  Salt Lake City
#  3  California   San Francisco
#  4    New-York        New York
#  5     Florida           Miami
#  6       Texas         Houston

# Concatenation can be performed by axis, 0:vertical, 1:horizontal

pd.concat([city_loc, city_pop], axis=1)

#    state            city        lat         lng  population           city       state
#  0    CA   San Francisco  37.781334 -122.416728         NaN            NaN         NaN
#  1    NY        New York  40.705649  -74.008344         NaN            NaN         NaN
#  2    FL           Miami  25.791100  -80.320733         NaN            NaN         NaN
#  3    OH       Cleveland  41.473508  -81.739791    808976.0  San Francisco  California
#  4    UT  Salt Lake City  40.755851 -111.896657   8363710.0       New York    New-York
#  5   NaN             NaN        NaN         NaN    413201.0          Miami     Florida
#  6   NaN             NaN        NaN         NaN   2242193.0        Houston       Texas

# Reindex the dataframe by city name, similar to a full outer join.
pd.concat([city_loc.set_index("city"), city_pop.set_index("city")], axis=1)

#                state        lat         lng  population       state
#  Cleveland         OH  41.473508  -81.739791         NaN         NaN
#  Houston          NaN        NaN         NaN   2242193.0       Texas
#  Miami             FL  25.791100  -80.320733    413201.0     Florida
#  New York          NY  40.705649  -74.008344   8363710.0    New-York
#  Salt Lake City    UT  40.755851 -111.896657         NaN         NaN
#  San Francisco     CA  37.781334 -122.416728    808976.0  California

city_loc.append(city_pop)  # Shorthand for vertical concatenation.

#               city        lat         lng  population       state
#  0   San Francisco  37.781334 -122.416728         NaN          CA
#  1        New York  40.705649  -74.008344         NaN          NY
#  2           Miami  25.791100  -80.320733         NaN          FL
#  3       Cleveland  41.473508  -81.739791         NaN          OH
#  4  Salt Lake City  40.755851 -111.896657         NaN          UT
#  3   San Francisco        NaN         NaN    808976.0  California
#  4        New York        NaN         NaN   8363710.0    New-York
#  5           Miami        NaN         NaN    413201.0     Florida
#  6         Houston        NaN         NaN   2242193.0       Texas
