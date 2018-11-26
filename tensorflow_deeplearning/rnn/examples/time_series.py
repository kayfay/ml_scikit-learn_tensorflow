# Python 2 and 3 support
from __future__ import division, print_function, unicode_literals

# Common Imports
import numpy as np
import os

# Graph Imports
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Config
PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


# Declare Functions
def save_fig(fig_id, tight_layout=True):
    if not os.path.exists('images'):
        os.makedirs('images')
    path = os.path.join(PROJECT_ROOT_DIR, 'images', fig_id + ".png")
    plt.savefig(path, format='png', dpi=300)


def time_series(t):
    """
    Calculates a timeseries

    Uses time point, t of sine waves at angle t
    """
    return t * np.sin(t) / 3 + 2 * np.sin(t * 5)


def next_batch(batch_size, n_steps):
    segment_num_samples = t_max - t_min - n_steps * resolution
    t0 = np.random.rand(batch_size, 1) * segment_num_samples
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(
        -1, n_steps, 1)


# Timeseries points
t_min, t_max = 0, 30
resolution = 0.1

# Graph axis points
num_samples = int((t_max - t_min) / resolution)
t = np.linspace(t_min, t_max, num_samples)

# Time point sequences
n_steps = 20
stop_value = 12.2 + resolution * (n_steps + 1)
num_samples = n_steps + 1
t_instance = np.linspace(12.2, stop_value, num_samples)

# Graph
plt.figure(figsize=(11, 4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(
    t_instance[:-1],
    time_series(t_instance[:-1]),
    "b-",
    linewidth=3,
    label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(
    t_instance[:-1],
    time_series(t_instance[:-1]),
    "bo",
    markersize=10,
    label="instance")
plt.plot(
    t_instance[1:],
    time_series(t_instance[1:]),
    "w*",
    markersize=10,
    label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")

save_fig("time_series_plot")
plt.show()

X_batch, y_batch = next_batch(1, n_steps)
print("1st    Input      Output")
print(np.c_[X_batch[0], y_batch[0]])
