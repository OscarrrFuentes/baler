import numpy as np
import argparse
import os

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--sep", action="store_true")

    args = parse.parse_args()
    return args

def load_data(args):
    cwd = os.getcwd()

    if args.sep:
        decomp_filename = os.path.join(cwd, "workspaces/MNIST/MNIST_project/output/decompressed_output/decompressed.npz")
        orig_filename = os.path.join(cwd, "workspaces/MNIST/data/non_outliers.npz")
    else:
        decomp_filename = os.path.join(cwd, "workspaces/MNIST/MNIST_project/output/decompressed_output/decompressed_with_outliers.npz")
        orig_filename = os.path.join(cwd, "workspaces/MNIST/data/outlier_order.npz")

    orig_file = np.load(orig_filename)
    decomp_file = np.load(decomp_filename)

    min_length = len(np.load(os.path.join(cwd, "workspaces/MNIST/data/non_outliers.npz"))["data"])
    print(f"min length: {min_length}")

    orig_file = {"data": orig_file["data"][:min_length], "names": orig_file["names"][:min_length]}
    decomp_file = {"data": decomp_file["data"][:min_length], "names": decomp_file["names"][:min_length]}
    
    if not np.all(orig_file["names"] == decomp_file["names"]):
        print(f"\n\n\n\nWrong filename used :(\norig_filename{orig_filename}\ndecomp_filename: {decomp_filename}\nargs.sep{args.sep}\n\n\n\n")
        print("Trying outlier_order.npz")
        orig_filename = os.path.join(cwd, "workspaces/MNIST/data/outlier_order.npz")
        orig_file = np.load(orig_filename)
        if np.all(orig_file["names"][:min_length] == decomp_file["names"]):
            print("That somehow worked")
        else:
            raise FileNotFoundError(f"{len(np.where(orig_file['names'][:min_length] != decomp_file['names']))} were wrong")

    return orig_file, decomp_file

def calculate_distance(orig_file, decomp_file, args):
    cwd = os.getcwd()
    digits = list(range(10))
    results = {}

    diff_arr = orig_file["data"].astype(np.float32) - decomp_file["data"].astype(np.float32)
    distances = np.array([np.sqrt(np.sum(image**2)) for image in diff_arr])
    print(f"\nlen(distances): {len(distances)}\n")

    for digit in digits:
        digit_distances = distances[np.where(orig_file["names"] == digit)]
        results[digit] = np.array((np.mean(digit_distances), np.std(digit_distances)))

        if args.sep:
            with open(os.path.join(cwd, f"outlier_proportion_testing/results/sep_{digit}.txt"), "a") as f:
                f.write(f"{results[digit][0]}, {results[digit][1]}\n")
        else:
            with open(os.path.join(cwd, f"outlier_proportion_testing/results/inc_{digit}.txt"), "a") as f:
                f.write(f"{results[digit][0]}, {results[digit][1]}\n")
    
    return results, digits

def main():
    args = parse_args()
    orig_file, decomp_file = load_data(args)
    if orig_file == -1:
        return -1
    results = calculate_distance(orig_file, decomp_file, args)
    if results == -1:
        return -1
    return 0
    

main()
