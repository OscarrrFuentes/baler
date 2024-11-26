import numpy as np
import argparse
import os


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--proportion", required=True)

    args = parse.parse_args()
    return args

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
    
    outlier_file_size = os.path.getsize(os.path.join(cwd, "workspaces/MNIST/data/outliers.npz")) * 4
    compressed_file_size = os.path.getsize(os.path.join(cwd, "workspaces/MNIST/MNIST_project/output/compressed_output/compressed.npz"))
    

    with open(os.path.join("/gluster/home/ofrebato/baler/outlier_proportion_testing/full_results.txt"), "a") as f:
        f.write(f"{args.proportion}, {outlier_file_size}, {compressed_file_size}, {np.mean(perc_diffs)}, {np.std(perc_diffs)}\n")
        print("Written results to file")
        print(outlier_file_size, compressed_file_size)
    
    return 0

def main():
    args = parse_args()
    return mean_perc_diff(args)

main()