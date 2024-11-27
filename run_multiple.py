import numpy as np
from scipy.stats import chi2
import subprocess
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import argparse
import os
import sys
from pathlib import Path

DEFUALT_N = 10
CWD = os.getcwd()

config_path = Path(CWD+"/workspaces/MNIST/MNIST_project/config")
helper_path = Path(CWD+"/baler/modules")
baler_path = Path(CWD)

if str(config_path) not in sys.path:
    sys.path.append(str(config_path))
if str(helper_path) not in sys.path:
    sys.path.append(str(helper_path))
if str(baler_path) not in sys.path:
    sys.path.append(str(baler_path))

from baler.modules.helper import Config
import MNIST_project_config as pc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--repetitions",
        type=int,
        default=DEFUALT_N,
        help="Specify the amount of repetitions (default: 10)",
    )
    parser.add_argument(
        "--fit",
        default="mean",
        choices=("mean", "chi2"),
        help="Specify how you want to fit the histogram of differences, choice between 'mean' and 'chi2' (default: mean)",
    )
    parser.add_argument(
        "-s", "--savename", default = "run_multiple_results/results.npz", help="Results filename"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose output"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1000, help="Specify batch size (default: 1000)"
    )
    args = parser.parse_args()
    return args


def load_data():

    orig_file = np.load(CWD+"/workspaces/MNIST/data/mnist_combined.npz")
    decomp_file = np.load(
        CWD+"/workspaces/MNIST/MNIST_project/output/decompressed_output/decompressed.npz"
    )

    if not np.all(orig_file["data"].shape == decomp_file["data"].shape):
        orig_file = np.load(CWD+"/workspaces/MNIST/data/non_outliers.npz")
    elif not np.all(orig_file["names"] == decomp_file["names"]):
        orig_file = np.load(CWD+"/workspaces/MNIST/data/outlier_order.npz")

    orig_data = orig_file["data"].astype(np.float32)
    decomp_data = decomp_file["data"].astype(np.float32)
    names = orig_file["names"][: len(orig_data)]

    if not (len(orig_data) == len(decomp_data) == len(names)):
        print(len(orig_data), len(decomp_data), len(names))
        raise ValueError("AHHHHHHHHHHHH")

    return orig_data, decomp_data, names


def get_numbers(names):
    zeros = np.where(names == 0)[0]
    ones = np.where(names == 1)[0]
    twos = np.where(names == 2)[0]
    threes = np.where(names == 3)[0]
    fours = np.where(names == 4)[0]
    fives = np.where(names == 5)[0]
    sixes = np.where(names == 6)[0]
    sevens = np.where(names == 7)[0]
    eights = np.where(names == 8)[0]
    nines = np.where(names == 9)[0]

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

    return numbers


def run_chi2_fit(error_arr, numbers, result_dict, df_set):
    for number in numbers:
        errors = error_arr[number[0]]
        trim_error_arr = errors[np.where(errors > 0)]
        _, loc, scale = chi2.fit(trim_error_arr)
        for _ in range(5):
            _, loc, scale = chi2.fit(trim_error_arr, f0=df_set[number[1]], loc=loc, scale=scale)
        result_dict[number[1]] = np.vstack((result_dict[number[1]], np.array([loc, scale])))
    return result_dict


def run_mean_fit(error_arr, numbers, result_dict):
    for number in numbers:
        errors = error_arr[number[0]]
        trim_error_arr = errors[np.where(errors > 0)]
        mean = np.mean(trim_error_arr)
        std = np.std(trim_error_arr)
        result_dict[number[1]] = np.vstack((result_dict[number[1]], np.array([mean, std])))
    return result_dict


def run_baler():
    with open("job.all_output", "a", encoding="utf-8") as fout, open("job.err", "a", encoding="utf-8") as ferr:
        res = subprocess.run(
            ["poetry", "run", "baler", "--project", "MNIST", "MNIST_project", "--mode", "train"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        fout.write(res.stdout.decode("utf-8"))
        fout.write(res.stderr.decode("utf-8"))
        
        res = subprocess.run(
            ["poetry", "run", "baler", "--project", "MNIST", "MNIST_project", "--mode", "compress"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        fout.write(res.stdout.decode("utf-8"))
        fout.write(res.stderr.decode("utf-8"))
        
        res = subprocess.run(
            ["poetry", "run", "baler", "--project", "MNIST", "MNIST_project", "--mode", "decompress"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        fout.write(res.stdout.decode("utf-8"))
        fout.write(res.stderr.decode("utf-8"))


def print_time(start, count, total):
    elapsed_time = time.time() - start
    time_left = elapsed_time * (total / count) - elapsed_time
    elapsed_time_delta = timedelta(seconds=round(elapsed_time))
    time_left_delta = timedelta(seconds=round(time_left))
    print(
        f"\n{count} completed\nTime taken: {elapsed_time_delta}\nEstimated time remaining: {time_left_delta}\n"
    )


def get_df_set(diff_arr, numbers):
    df_set = {}
    error_arr = np.array([np.sum(image**2) for image in diff_arr])
    for number in numbers:
        errors = error_arr[number[0]]
        errors = errors[np.where(errors > 0)]
        df, loc, scale = chi2.fit(errors)
        for _ in range(5):
            if df >= 100:
                df = 6
            df, loc, scale = chi2.fit(errors, df, loc=loc, scale=scale)
        df_set[number[1]] = df
    return df_set


def get_results(n, fit_type, verbose, batch_size):
    with open(CWD+"/workspaces/MNIST/MNIST_project/config/MNIST_project_config.py", "a") as f:
        f.write(f"\n    c.batch_size = {batch_size}")

    digits_spelled = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]

    separate_outliers = {digit: np.empty((0, 2)) for digit in digits_spelled}
    with_outliers = {digit: np.empty((0, 2)) for digit in digits_spelled}

    start = time.time()

    with open(CWD+"/workspaces/MNIST/MNIST_project/config/MNIST_project_config.py", "a") as f:
        f.write("\n    c.separate_outliers = True")

    start_run = True
    df_dict = {}
    outlier_length = 0
    print("\nFinding outliers...\n")

    for i in range(n):
        run_baler()
        orig_data, decomp_data, names = load_data()
        diff_arr = orig_data.astype(np.float32) - decomp_data.astype(np.float32)
        numbers = get_numbers(names)

        if fit_type == "chi2":
            # if start_run:
            #     df_dict = get_df_set(diff_arr, numbers)
            #     start_run = False
            errors = np.array([np.sqrt(np.sum(image**2)) for image in diff_arr])
            separate_outliers = run_chi2_fit(errors, numbers, separate_outliers, df_dict)
        elif fit_type == "mean":
            errors = np.array([np.sqrt(np.sum(image**2)) for image in diff_arr])
            separate_outliers = run_mean_fit(errors, numbers, separate_outliers)
        else:
            raise ValueError("fit can only be mean or chi2")
        if start_run:
            with open(CWD+"/workspaces/MNIST/MNIST_project/config/MNIST_project_config.py", "a") as f:
                f.write("\n    c.separate_outliers = False")
            outlier_length = len(errors)
            print(f"\nNumber of outliers: {70000 - outlier_length}\n")
            start += (time.time() - start)/2 #Had to do two trainings for the first one
            start_run=False

        print_time(start, i + 1, 2 * n)
        if verbose:
            print("Results")
            for key in separate_outliers.keys():
                print(key, f"Mean: {separate_outliers[key][i][0]:.2f}\tStd: {separate_outliers[key][i][1]:.2f}")

    with open(CWD+"/workspaces/MNIST/MNIST_project/config/MNIST_project_config.py", "a") as f:
        f.write(f"\n    c.input_path = \"{CWD}/workspaces/MNIST/data/outlier_order.npz\"")
        f.write("\n    c.separate_outliers=False")

    print("\nSeparating outliers complete!\nRunning including outliers...\n")

    for i in range(n):
        run_baler()
        orig_data, decomp_data, names = load_data()
        diff_arr = (orig_data - decomp_data).astype(np.float32)
        numbers = get_numbers(names)
        if fit_type == "chi2":
            if start_run:
                df_dict = get_df_set(diff_arr, numbers)
                start_run = False
            errors = np.array([np.sqrt(np.sum(image**2)) for image in diff_arr])
            with_outliers = run_chi2_fit(errors, numbers, with_outliers, df_dict)
        elif fit_type == "mean":
            errors = np.array([np.sqrt(np.sum(image**2)) for image in diff_arr])
            with_outliers = run_mean_fit(errors, numbers, with_outliers)
        else:
            raise ValueError("fit can only be mean or chi2")

        print_time(start, i + 1 + n, 2 * n)
        if verbose:
            print("Results")
            for key in with_outliers.keys():
                print(key, f"Mean: {with_outliers[key][i][0]:.2f}\tStd: {with_outliers[key][i][1]:.2f}")

    with open(CWD+"/workspaces/MNIST/MNIST_project/config/MNIST_project_config.py", "r+") as f:
        lines = f.readlines()
        f.write("\n"+lines[1])

    return separate_outliers, with_outliers, numbers


def plot_results(separate_outliers, with_outliers, numbers, n):
    config=Config
    pc.set_config(config)

    fig, ax = plt.subplots(figsize=(12, 6))
    start=True
    for key in separate_outliers.keys():
        if start:
            ax.errorbar([key]*len(separate_outliers[key]), separate_outliers[key][:,0], yerr=separate_outliers[key][:,1], fmt="bx", label="Separating outliers", capsize=3, elinewidth=1, alpha=0.3)
            ax.errorbar([key]*len(with_outliers[key]), with_outliers[key][:,0], yerr=with_outliers[key][:,1], fmt="rx", label = "Including outliers", capsize=3, elinewidth=1, alpha=0.3)
            start=False
        else:
            ax.errorbar([key]*len(separate_outliers[key]), separate_outliers[key][:,0], yerr=separate_outliers[key][:,1], fmt="bx", capsize=3, elinewidth=1, alpha=0.3)
            ax.errorbar([key]*len(with_outliers[key]), with_outliers[key][:,0], yerr=with_outliers[key][:,1], fmt="rx", capsize=3, elinewidth=1, alpha=0.3)
    ax.legend()
    ax.set_xlabel("Digit")
    ax.set_ylabel("Mean distance")
    ax.grid(color="silver", alpha=0.5)
    fig.suptitle(f"Baler improvement from separating outliers. Epochs={config.epochs}, Batch size={config.batch_size}, Repetitions={n}")
    plt.savefig("All_mean_distances.png", dpi=600)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    perc_diffs = []
    for key in separate_outliers.keys():
        mean_including = np.mean(with_outliers[key][:,0])
        std_including = np.std(with_outliers[key][:,0])
        mean_separated = np.mean(separate_outliers[key][:,0])
        std_separated = np.std(separate_outliers[key][:,0])

        ax.errorbar(key, mean_separated, yerr=std_separated, fmt="bx", capsize=3, elinewidth=1)
        ax.errorbar(key, mean_including, yerr=std_including, fmt="rx", capsize=3, elinewidth=1)
        perc_diff = (mean_including - mean_separated) * 100 / mean_including
        perc_diffs.append(perc_diff)
        y_pos = min([mean_including, mean_separated]) - max([std_including, std_separated]) - 5
        ax.text(key, y_pos, f"{perc_diff:.2f}%", ha="center", va="top")
    ax.errorbar(key, np.mean(separate_outliers[key][:,0]), yerr=np.std(separate_outliers[key][:,0]), fmt="bx", label="Separating outliers", capsize=3, elinewidth=1)
    ax.errorbar(key, np.mean(with_outliers[key][:,0]), yerr=np.std(with_outliers[key][:,0]), fmt="rx", label="Including outliers", capsize=3, elinewidth=1)
    ax.legend()
    ax.set_xlabel("Digit")
    ax.set_ylabel("Mean distance")
    ax.grid(color="silver", alpha=0.5)
    fig.suptitle(f"Mean Baler improvement from separating outliers. Epochs={config.epochs}, Batch size={config.batch_size}, Repetitions={n}\nMean improvement = {np.mean(perc_diffs):.3f}% $\pm$ {np.std(perc_diffs):.3f}%", fontsize=18)
    plt.savefig("Mean_mean_distances.png", dpi=600)
    plt.close()

    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(26, 12))
    for key, ax in zip(separate_outliers.keys(), axs.flatten()):
        n_sep, bins_sep, patches_sep = ax.hist(separate_outliers[key][:,0], bins=int(np.ceil(np.log2(n) + 1)), color="royalblue", alpha=0.7, label="Separate")
        n_with, bins_with, patches_with = ax.hist(with_outliers[key][:,0], bins=int(np.ceil(np.log2(n) + 1)), color="orange", alpha=0.7, label="Including")
        ax.set_xlabel("Mean distance", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.set_title(key, fontsize=16)

        mean_sep = np.mean(separate_outliers[key][:,0])
        std_sep = np.std(separate_outliers[key][:,0])
        ax.axvline(x=mean_sep, color="royalblue", linestyle="dashed")
        ax.axvspan(mean_sep-std_sep, mean_sep+std_sep, alpha=0.15, color="royalblue")

        mean_inc = np.mean(with_outliers[key][:,0])
        std_inc = np.std(with_outliers[key][:,0])
        ax.axvline(x=mean_inc, color="darkorange", linestyle="dashed")
        ax.axvspan(mean_inc-std_inc, mean_inc+std_inc, alpha=0.15, color="darkorange")
        ax.grid(color="silver", alpha=0.5)

        ax.legend()

    fig.suptitle(f"Distributions of mean distances from original images. Epochs={config.epochs}, Batch size={config.batch_size}, Repetitions={n}", fontsize=24)
    plt.savefig("Hist_mean_distances.png")

def main():
    with open("job.all_output", "w", encoding="utf-8") as fout:
        fout.write("Starting multiple runs...")

    args = parse_args()
    print(f"Starting run...\nRepetitions: {args.repetitions}" + f"\nFit type: {args.fit}"\
          +f"\nBatch size: {args.batch_size}")

    separate_outliers, with_outliers, numbers = get_results(args.repetitions, args.fit, args.verbose, args.batch_size)
    print("    separate:::with")

    for i in separate_outliers.keys():
        sep_mean = np.mean(separate_outliers[i])
        with_mean = np.mean(with_outliers[i])
        print(f"{i}: {sep_mean:.2f}:::{with_mean:.2f}")

    print(f"\nSaving results to {args.savename}...\n")

    if not os.path.exists("run_multiple_results"):
        os.mkdir("run_multiple_results")
    np.savez_compressed(
        args.savename, separate=separate_outliers, including=with_outliers
    )

    plot_results(separate_outliers, with_outliers, numbers, args.repetitions)



main()
