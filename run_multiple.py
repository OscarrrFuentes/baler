import numpy as np
from scipy.stats import chi2
import subprocess
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import argparse
import os


DEFUALT_N = 10


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
        "-s", "--save", action="store_true", help="Flag whether to save the results or not"
    )
    args = parser.parse_args()
    return args


def load_data():

    orig_file = np.load("workspaces/MNIST/data/mnist_combined.npz")
    decomp_file = np.load(
        "workspaces/MNIST/MNIST_project/output/decompressed_output/decompressed.npz"
    )

    if not np.all(orig_file["names"] == decomp_file["names"]):
        orig_file = np.load("workspaces/MNIST/data/mnist_combined_outlier_order.npz")

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
    subprocess.run(
        ["poetry", "run", "baler", "--project", "MNIST", "MNIST_project", "--mode", "train"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ["poetry", "run", "baler", "--project", "MNIST", "MNIST_project", "--mode", "compress"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ["poetry", "run", "baler", "--project", "MNIST", "MNIST_project", "--mode", "decompress"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


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


def get_results(n, fit_type):
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

    with open("workspaces/MNIST/MNIST_project/config/MNIST_project_config.py", "a") as f:
        f.write("\n    c.separate_outliers = True")

    start_run = True
    df_dict = {}

    for i in range(n):
        run_baler()
        orig_data, decomp_data, names = load_data()
        diff_arr = (orig_data - decomp_data).astype(np.float32)
        numbers = get_numbers(names)

        if fit_type == "chi2":
            if start_run:
                df_dict = get_df_set(diff_arr, numbers)
                start_run = False
            errors = np.array([np.sum(image**2) for image in diff_arr])
            separate_outliers = run_chi2_fit(errors, numbers, separate_outliers, df_dict)
        elif fit_type == "mean":
            errors = np.array([np.sum(image**2) for image in diff_arr])
            separate_outliers = run_mean_fit(errors, numbers, separate_outliers)
        else:
            raise ValueError("fit can only be mean or chi2")

        print_time(start, i + 1, 2 * n)

    with open("workspaces/MNIST/MNIST_project/config/MNIST_project_config.py", "a") as f:
        f.write("\n    c.separate_outliers = False")

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
            errors = np.array([np.sum(image**2) for image in diff_arr])
            with_outliers = run_chi2_fit(errors, numbers, with_outliers, df_dict)
        elif fit_type == "mean":
            errors = np.array([np.sum(image**2) for image in diff_arr])
            with_outliers = run_mean_fit(errors, numbers, with_outliers)
        else:
            raise ValueError("fit can only be mean or chi2")

        print_time(start, i + 1 + n, 2 * n)

    return separate_outliers, with_outliers, numbers


def plot_results(separate_outliers, with_outliers, numbers):
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(22, 10), sharey=True, sharex=True)
    for number, ax in zip(numbers, axs.flatten()):
        x1 = separate_outliers[number[1]]
        x2 = with_outliers[number[1]]
        n1, bins1 = np.histogram(x1, bins=len(x1) // 4)
        n2, bins2 = np.histogram(x2, bins=len(x2) // 4)
        ax.plot(bins1[:-1], n1, "r-", label="Separate")
        ax.plot(bins2[:-1], n2, "b-", label="With")
        ax.set_xlabel("Chi2 loc", fontsize=16)
        ax.set_ylabel("Count", fontsize=16)
        ax.set_title(number[1], fontsize=20)
        ax.legend()
    fig.suptitle("Difference of Chi2 location", fontsize=24)
    plt.savefig("all_chi_distributions.png", dpi=900)
    plt.close()


def main():
    args = parse_args()
    print(f"Starting run...\nRepetitions: {args.repetitions}" + f"\nFit type: {args.fit}")

    separate_outliers, with_outliers, numbers = get_results(args.repetitions, args.fit)
    print("    separate:::with")

    for i in separate_outliers.keys():
        sep_mean = np.mean(separate_outliers[i])
        with_mean = np.mean(with_outliers[i])
        print(f"{i}: {sep_mean:.2f}:::{with_mean:.2f}")

    if args.save:
        print("\nSaving results to run_multiple_results/results.npz...\n")

        if not os.path.exists("run_multiple_results"):
            os.mkdir("run_multiple_results")
        np.savez_compressed(
            "run_multiple_results/results.npz", separate=separate_outliers, including=with_outliers
        )

    plot_results(separate_outliers, with_outliers, numbers)


main()
