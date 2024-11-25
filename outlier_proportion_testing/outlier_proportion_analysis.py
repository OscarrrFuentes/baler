import numpy as np
import argparse
import os


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--proportion", required=True)

    args = parse.parse_args()
    return args

def human_readable_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024

def mean_perc_diff(args):
    perc_diffs = []
    cwd = os.getcwd()
    child_wd = os.path.join(cwd, "outlier_proportion_testing")

    for digit in list(range(10)):
        inc_data = np.loadtxt(os.path.join(child_wd, "results", f"inc_{digit}.txt"), dtype=np.float32, delimiter=",")
        sep_data = np.loadtxt(os.path.join(child_wd, "results", f"sep_{digit}.txt"), dtype=np.float32, delimiter=",")
        mean_including = np.mean(inc_data[:,0])
        mean_separated = np.mean(sep_data[:,0])
        perc_diffs.append((mean_including - mean_separated) * 100/mean_including)
    
    file_size = os.path.getsize(os.path.join(cwd, "workspaces/MNIST/data/outliers.npz"))
    file_size = human_readable_size(file_size*4)

    with open(os.path.join(child_wd, "full_results.txt"), "a") as f:
        f.write(f"{args.proportion}, {file_size}, {np.mean(perc_diffs)}, {np.std(perc_diffs)}\n")
        print("Written results to file")
        print(file_size)
    
    return 0

def main():
    args = parse_args()
    return mean_perc_diff(args)

main()