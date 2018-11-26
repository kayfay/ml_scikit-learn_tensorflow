# Timestamps
# * Peroid representation and frequencies, 2016Q3 and "monthly".
# * Convert periods from/to actual timestamps.
# * Resample data and aggregate values.
# * Handle timezones.

import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Create utility functions for saving the plot.

# Directory Config
PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


# Declare Functions
def image_path(fig_id):
    save_dir = os.path.join(PROJECT_ROOT_DIR, 'images')
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, fig_id + ".png")


def save_fig(fig_id, tight_layout=True):
    print("Saving ", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id), format='png', dpi=300)


# Creating time series (ranges of time), indexes/frequencies
# containing one date time per 12 hours, peroid per hour.
dates = pd.date_range('2016/10/29 5:30pm', periods=12, freq='H')

#  DatetimeIndex(['2016-10-29 17:30:00', '2016-10-29 18:30:00',
#                 '2016-10-29 19:30:00', '2016-10-29 20:30:00',
#                 '2016-10-29 21:30:00', '2016-10-29 22:30:00',
#                 '2016-10-29 23:30:00', '2016-10-30 00:30:00',
#                 '2016-10-30 01:30:00', '2016-10-30 02:30:00',
#                 '2016-10-30 03:30:00', '2016-10-30 04:30:00'],
#                dtype='datetime64[ns]', freq='H')

# Create a Series using the DatetimeIndex for labels of the time series.
temperatures = [4.4, 5.1, 6.1, 6.2, 6.1, 6.1, 5.7, 5.2, 4.7, 4.1, 3.9, 3.5]
labeled_series = pd.Series(temperatures, dates)

#  2016-10-29 17:30:00    4.4
#  2016-10-29 18:30:00    5.1
#  2016-10-29 19:30:00    6.1
#  2016-10-29 20:30:00    6.2
#  2016-10-29 21:30:00    6.1
#  2016-10-29 22:30:00    6.1
#  2016-10-29 23:30:00    5.7
#  2016-10-30 00:30:00    5.2
#  2016-10-30 01:30:00    4.7
#  2016-10-30 02:30:00    4.1
#  2016-10-30 03:30:00    3.9
#  2016-10-30 04:30:00    3.5
#  Freq: H, dtype: float64

# Plot
labeled_series.plot(kind="bar")
plt.grid(True)
plt.show()
save_fig('labeled_timeseries_histogram')

# Resampling by specifying a new frequency to permeate a
# DatetimeIndexResampler object. (downsampling, reducing data)
labeled_time_series_freq_2H = labeled_series.resample("2H")

#  DatetimeIndexResampler[
#      freq = <2 * Hours > , axis = 0,
#      closed = left, label = left, convention = start, base = 0]

# To compute pairs of consecutive hours for
# the resampling operation use the mean.
labeled_time_series_freq_2H.mean().plot(kind="bar")
plt.show()
save_fig('resample')

# Upsampling (increase frequency) but with holes in data
time_series_freq_15min = labeled_series.resample("15Min").mean()
time_series_freq_15min.head(n=10)

#  2016-10-29 17:30:00    4.4
#  2016-10-29 17:45:00    NaN
#  2016-10-29 18:00:00    NaN
#  2016-10-29 18:15:00    NaN
#  2016-10-29 18:30:00    5.1
#  2016-10-29 18:45:00    NaN
#  2016-10-29 19:00:00    NaN
#  2016-10-29 19:15:00    NaN
#  2016-10-29 19:30:00    6.1
#  2016-10-29 19:45:00    NaN

# Interpolation to fill gaps, e.g., linear interpolation, cubic interpolation
time_series_freq_15min = labeled_series.resample("15Min").interpolate(
    method="cubic")
time_series_freq_15min.head(n=10)

#  2016-10-29 17:30:00    4.400000
#  2016-10-29 17:45:00    4.452911
#  2016-10-29 18:00:00    4.605113
#  2016-10-29 18:15:00    4.829758
#  2016-10-29 18:30:00    5.100000
#  2016-10-29 18:45:00    5.388992
#  2016-10-29 19:00:00    5.669887
#  2016-10-29 19:15:00    5.915839
#  2016-10-29 19:30:00    6.100000
#  2016-10-29 19:45:00    6.203621
#  Freq: 15T, dtype: float64

labeled_series.plot(label="Peroid: 1 hour")
time_series_freq_15min.plot(label="Peroid: 15 minutes")
plt.legend()
plt.show()
save_fig('interpolated_comparison_plot')

# Timezones in datetime can be set with timezone specifying methods
# datetimes refer to UTC, -/+ hour are appended.
labeled_series_ny = labeled_series.tz_localize("America/New_York")

#  2016-10-29 17:30:00-04:00    4.4
#  2016-10-29 18:30:00-04:00    5.1
#  2016-10-29 19:30:00-04:00    6.1
#  2016-10-29 20:30:00-04:00    6.2
#  2016-10-29 21:30:00-04:00    6.1
#  2016-10-29 22:30:00-04:00    6.1
#  2016-10-29 23:30:00-04:00    5.7
#  2016-10-30 00:30:00-04:00    5.2
#  2016-10-30 01:30:00-04:00    4.7
#  2016-10-30 02:30:00-04:00    4.1
#  2016-10-30 03:30:00-04:00    3.9
#  2016-10-30 04:30:00-04:00    3.5
#  Freq: H, dtype: float64

# Timezones can also be converted.
labeled_series_paris = labeled_series_ny.tz_convert("Europe/Paris")

#  2016-10-29 23:30:00+02:00    4.4
#  2016-10-30 00:30:00+02:00    5.1
#  2016-10-30 01:30:00+02:00    6.1
#  2016-10-30 02:30:00+02:00    6.2  # Note double entries for UTC
#  2016-10-30 02:30:00+01:00    6.1  # caused by daylight savings time.
#  2016-10-30 03:30:00+01:00    6.1
#  2016-10-30 04:30:00+01:00    5.7
#  2016-10-30 05:30:00+01:00    5.2
#  2016-10-30 06:30:00+01:00    4.7
#  2016-10-30 07:30:00+01:00    4.1
#  2016-10-30 08:30:00+01:00    3.9
#  2016-10-30 09:30:00+01:00    3.5
#  Freq: H, dtype: float64

# When the timezone is updated revertered to a naive representation
# the datetimes become ambigious.
labeled_series_paris_naive = labeled_series_paris.tz_localize(None)
labeled_series_paris_naive

#  2016-10-29 23:30:00    4.4
#  2016-10-30 00:30:00    5.1
#  2016-10-30 01:30:00    6.1
#  2016-10-30 02:30:00    6.2
#  2016-10-30 02:30:00    6.1
#  2016-10-30 03:30:00    6.1
#  2016-10-30 04:30:00    5.7
#  2016-10-30 05:30:00    5.2
#  2016-10-30 06:30:00    4.7
#  2016-10-30 07:30:00    4.1
#  2016-10-30 08:30:00    3.9
#  2016-10-30 09:30:00    3.5
#  Freq: H, dtype: float64

# Localizing a new timezone makes an error.
try:
    labeled_series_paris_naive.tz_localize("Europe/Paris")
except Exception as e:
    print(type(e))
    print(e)

#  <class 'pytz.exceptions.AmbiguousTimeError'>
#  Cannot infer dst time from '2016-10-30 02:30:00', try using the 'ambiguous' argument

# DST can be adjusted by setting the ambiguous to infer.
labeled_series_paris_naive.tz_localize("Europe/Paris", ambiguous="infer")

#  2016-10-29 23:30:00+02:00    4.4
#  2016-10-30 00:30:00+02:00    5.1
#  2016-10-30 01:30:00+02:00    6.1
#  2016-10-30 02:30:00+02:00    6.2
#  2016-10-30 02:30:00+01:00    6.1
#  2016-10-30 03:30:00+01:00    6.1
#  2016-10-30 04:30:00+01:00    5.7
#  2016-10-30 05:30:00+01:00    5.2
#  2016-10-30 06:30:00+01:00    4.7
#  2016-10-30 07:30:00+01:00    4.1
#  2016-10-30 08:30:00+01:00    3.9
#  2016-10-30 09:30:00+01:00    3.5
#  Freq: H, dtype: float64
