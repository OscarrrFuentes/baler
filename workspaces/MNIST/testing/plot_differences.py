import numpy as np
import matplotlib.pyplot as plt

orig_data = np.load("/Users/oscarfuentes/baler/workspaces/MNIST/data/mnist_combined.npz")
decomp_data = np.load(
    "/Users/oscarfuentes/baler/workspaces/MNIST/MNIST_project/output/decompressed_output/decompressed.npz"
)

zeros = np.where(orig_data["names"] == 0)[0]
ones = np.where(orig_data["names"] == 1)[0]
twos = np.where(orig_data["names"] == 2)[0]
threes = np.where(orig_data["names"] == 3)[0]
fours = np.where(orig_data["names"] == 4)[0]
fives = np.where(orig_data["names"] == 5)[0]
sixes = np.where(orig_data["names"] == 6)[0]
sevens = np.where(orig_data["names"] == 7)[0]
eights = np.where(orig_data["names"] == 8)[0]
nines = np.where(orig_data["names"] == 9)[0]

diff_arr = orig_data["data"] - decomp_data["data"]
error_list = []

counter = 0
for image in diff_arr:
    error = 0
    for line in image:
        error += np.sqrt(np.sum(line**2))
    error_list.append(error)
    counter += 1
    if counter % 10000 == 0:
        print(f"{counter} images processed")

np_error_list = np.array(error_list)

print("Plotting all...")
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(np_error_list, bins=len(np_error_list) // 20)
ax.set_title("Root sum square of the differences for every image")
ax.set_xlabel("Error value")
ax.set_ylabel("Count")

plt.savefig("workspaces/MNIST/testing/output_graphs/differences/all_differences.png", dpi=600)
plt.close()

print("Plotted all\n")

numbers = [
    [zeros, "zero"],
    [ones, "one"],
    [twos, "two"],
    [threes, "three"],
    [fours, "four"],
    [fives, "five"],
    [sixes, "six"],
    [sevens, "seven"],
    [eights, "eight"],
    [nines, "nine"],
]

print("Plotting individual number graphs...\n")

for number in numbers:
    fig, ax = plt.subplots(figsize=(12, 6))

    n, bins, patches = ax.hist(
        np_error_list[np.array(number[0])], bins=len(np_error_list[np.array(number[0])]) // 20
    )
    ax.set_title('Root sum square of the differences for every "' + str(number[1]) + '" image')
    ax.set_xlabel("Error value")
    ax.set_ylabel("Count")
    plt.savefig(
        "workspaces/MNIST/testing/output_graphs/differences/" + str(number[1]) + "_differences.png",
        dpi=600,
    )
    plt.close()
    print("Plotted " + str(number[1]) + f" graph, peak at {bins[np.where(n == max(n))][0]}")
