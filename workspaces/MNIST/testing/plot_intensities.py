import numpy as np
import matplotlib.pyplot as plt
import time

import matplotlib as mpl
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter

mpl.rcParams["agg.path.chunksize"] = 10000


# Olly's custom colourmap that he's really proud of
_synthwave = LinearSegmentedColormap.from_list(
    "_synthwave",
    [
        (0, "#ffffff"),
        (1e-20, "#2f4b7c"),
        (0.2, "#665191"),
        (0.4, "#a05195"),
        (0.6, "#d45087"),
        (0.8, "#ff7c43"),
        (1, "#ffa600"),
    ],
    N=256,
)


def using_mpl_scatter_density(fig, x, y, title, filename):
    ax = fig.add_subplot(projection="scatter_density")

    density = ax.scatter_density(x, y, dpi=30, cmap=_synthwave)

    fig.colorbar(density, label="Number of points per pixel")
    # ax.set_yscale("log")
    ax.set_ylim(-100, 300)

    ax.set_ylabel("Whiteness (0-255)", fontname="Times New Roman", fontsize=14, weight="bold")
    ax.set_xlabel("Position", fontname="Times New Roman", fontsize=14, weight="bold")
    # ax.set_xlim(0, 600)
    ax.set_title(title, fontname="Times New Roman", fontsize=18, weight="bold")

    plt.savefig(filename, dpi=600)
    plt.close(fig)
    print("saved to ", filename)


data = np.load("../MNIST_project/output/decompressed_output/decompressed.npz")
zeros = np.where(data["names"] == 0)[0]

counter = 0
plot_data = np.empty((0, 2))
for i in data["data"][zeros[:10]]:
    print(counter, "arrays completed")
    arr = np.concatenate(i)
    for count, val in enumerate(arr):
        if val != 0 or val == 0:
            plot_data = np.vstack((plot_data, (count, val)))
    counter += 1
print("\nDecompressed data completed\n")

data_compare = np.load("../data/mnist_combined.npz")
zeros_compare = np.where(data_compare["data"] == 0)[0]

counter = 0
plot_data_compare = np.empty((0, 2))
for i in data_compare["data"][zeros_compare[:10]]:
    print(counter, "arrays completed")
    arr = np.concatenate(i)
    for count, val in enumerate(arr):
        if val != 0 or val == 0:
            plot_data_compare = np.vstack((plot_data_compare, (count, val)))
    counter += 1

print("\nOriginal data completed")

decomp_title = "Non-black pixels at certain positions for zeros (Decompressed data)"
decomp_filename = "output_graphs/decompressed_zero_values_intensity.png"
orig_title = "Non-black pixels at certain positions for zeros (Original data)"
orig_filename = "output_graphs/original_zero_values_intensity.png"

print("\n\n", len(plot_data[:, 0]), "\n", len(plot_data_compare[:, 1]))

fig = plt.figure(figsize=(12, 6))
using_mpl_scatter_density(fig, plot_data[:, 0], plot_data[:, 1], decomp_title, decomp_filename)

fig = plt.figure(figsize=(12, 6))
using_mpl_scatter_density(
    fig, plot_data_compare[:, 0], plot_data_compare[:, 1], orig_title, orig_filename
)
