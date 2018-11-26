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

# Categories or factors, fore example 1 for female 2 for male
# A for Good, B for Average, C for Bad
city_eco = city_pop.copy()
city_eco["eco_code"] = [17, 17, 34, 20]  # Create a meaningless column.
city_eco

#     population           city       state  eco_code
#  3      808976  San Francisco  California        17
#  4     8363710       New York    New-York        17
#  5      413201          Miami     Florida        34
#  6     2242193        Houston       Texas        20

# Set type to category and give names for each code.
city_eco["economy"] = city_eco["eco_code"].astype('category')
city_eco["economy"].cat.categories

#  Int64Index([17, 20, 34], dtype='int64')

city_eco["economy"].cat.categories = ["Finance", "Energy", "Tourism"]
city_eco

#     population           city       state  eco_code  economy
#  3      808976  San Francisco  California        17  Finance
#  4     8363710       New York    New-York        17  Finance
#  5      413201          Miami     Florida        34  Tourism
#  6     2242193        Houston       Texas        20   Energy

# Sort based on acategorical order.
city_eco.sort_values(by="economy", ascending=False)

#     population           city       state  eco_code  economy
#  5      413201          Miami     Florida        34  Tourism
#  6     2242193        Houston       Texas        20   Energy
#  4     8363710       New York    New-York        17  Finance
#  3      808976  San Francisco  California        17  Finance
