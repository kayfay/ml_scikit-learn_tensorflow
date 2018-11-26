import pandas as pd

# PeroidIndex for quarters in date 2010-2020
quarters = pd.period_range('2016Q1', periods=8, freq='Q')
quarters

#  PeriodIndex(['2016Q1', '2016Q2', '2016Q3', '2016Q4', '2017Q1', '2017Q2',
#               '2017Q3', '2017Q4'],
#              dtype='period[Q-DEC]', freq='Q-DEC')

# Shift peroids N times the PeroidIndex's frequency
quarters + 3

#  PeriodIndex(['2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4', '2018Q1',
#               '2018Q2', '2018Q3'],
#              dtype='period[Q-DEC]', freq='Q-DEC')

# Change frequency of PeroidIndex, lengthen or shorten
# peroids (zooming) e.g., convert quarters to months.

quarters.asfreq("M")  # Splits from inner quartile.

#  PeriodIndex(
#              ['2016-03', '2016-06',
#               '2016-09', '2016-12',
#               '2017-03', '2017-06',
#               '2017-09', '2017-12'],
#              dtype='period[M]', freq='M')

# Zoom in at the start of a peroid.
quarters.asfreq("M", how="start")

#  PeriodIndex(['2016-01', '2016-04',
#               '2016-07', '2016-10',
#               '2017-01', '2017-04',
#               '2017-07', '2017-10'],
#              dtype='period[M]', freq='M')

# Zoom out. Yearly.
quarters.asfreq("A")

#  PeriodIndex(
#      ['2016', '2016',
#       '2016', '2016',
#       '2017', '2017',
#       '2017', '2017'],
#      dtype='period[A-DEC]', freq='A-DEC')

# Create a series with a PeroidIndex
quarterly_revenue = pd.Series(
    [300, 320, 290, 390, 320, 360, 310, 410], index=quarters)

quarterly_revenue

#  2016Q1    300
#  2016Q2    320
#  2016Q3    290
#  2016Q4    390
#  2017Q1    320
#  2017Q2    360
#  2017Q3    310
#  2017Q4    410
#  Freq: Q-DEC, dtype: int64

# Plot quarters
quarterly_revenue.plot(kind="line")
plt.show()

# Convert periods to timestamps, by first day 
# of each peroid or last hour, "D" / "H".
last_hours = quarterly_revenue.to_timestamp(how="end", freq="H")

#  2016-03-31 23:00:00    300
#  2016-06-30 23:00:00    320
#  2016-09-30 23:00:00    290
#  2016-12-31 23:00:00    390
#  2017-03-31 23:00:00    320
#  2017-06-30 23:00:00    360
#  2017-09-30 23:00:00    310
#  2017-12-31 23:00:00    410
#  Freq: Q-DEC, dtype: int64

last_hours.to_period()

#  2016Q1    300
#  2016Q2    320
#  2016Q3    290
#  2016Q4    390
#  2017Q1    320
#  2017Q2    360
#  2017Q3    310
#  2017Q4    410
#  Freq: Q-DEC, dtype: int64
