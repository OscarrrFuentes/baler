import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

data = np.load("../data/mnist_combined.npz")
zeros = np.where(data["names"] == 0)[0]
ones = np.where(data["names"] == 1)[0]
twos = np.where(data["names"] == 2)[0]
threes = np.where(data["names"] == 3)[0]
fours = np.where(data["names"] == 4)[0]
fives = np.where(data["names"] == 5)[0]
sixes = np.where(data["names"] == 6)[0]
sevens = np.where(data["names"] == 7)[0]
eights = np.where(data["names"] == 8)[0]
nines = np.where(data["names"] == 9)[0]

tot_fig = plt.figure()
tot_ax = tot_fig.add_subplot()
colors = [
    "b",
    "g",
    "r",
    "m",
    "gold",
    "orange",
    "cyan",
    "teal",
    "rebeccapurple",
    "mediumspringgreen",
]
names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
# for color_count, i in enumerate(data["data"][ones[:50]]):
#     arr = np.concatenate(i)
#     for count, val in enumerate(arr):
#         if val != 0:
#             # ax.scatter(count, val, s=5, color=colors[color_count], alpha=0.4, marker=".")
#             ax.scatter(count, val, s=5, alpha=0.4, marker=".")
numbers = [zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines]
for color_count, number in enumerate(numbers):
    print(f"Start {names[color_count]} data processing...")
    fig = plt.figure()
    ax = fig.add_subplot()
    results = []
    total = 0
    for i in data["data"][number]:
        arr = np.concatenate(i)
        length = len(arr)
        for count, val in enumerate(arr):
            if val != 0:
                total += 1
                results.append(count)

    print("Data processing complete")
    print(f"Start {names[color_count]} plotting...")

    tot_ax.hist(
        results, bins=length, color=colors[color_count], alpha=0.4, label=names[color_count]
    )

    ax.hist(results, bins=length)
    ax.set_title("Non-black pixels at certain positions for " + names[color_count])
    ax.set_ylabel("Count")
    ax.set_xlabel("Position")
    plt.figure(fig)
    plt.savefig("output_graphs/" + names[color_count] + "_distribution.png", dpi=600)
    plt.close(fig)
    print(f"{names[color_count]} plotting complete!" + "\n")

tot_ax.set_title("All distributions for each number")
tot_ax.set_xlabel("Position")
tot_ax.set_ylabel("Count")
tot_ax.legend()
plt.savefig("output_graphs/all_distributions.png", dpi=600)
plt.close(tot_fig)
print("\n\n" + f"Time taken {round(time.time() - start, 3)}s")
