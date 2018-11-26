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


# Plotting series using pandas and matplotlib.
temperatures = [4.4, 5.1, 6.1, 6.2, 6.1, 6.1, 5.7, 5.2, 4.7, 4.1, 3.9, 3.5]
s = pd.Series(temperatures, name="Temperature")
s.plot()
plt.show()
save_fig('series_temperature_plot')
