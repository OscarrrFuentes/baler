import numpy as np
import argparse

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--values", required=True)
    parse.add_argument("--batch_size", type=int, required=True)
    parse.add_argument("--sep", action="store_true")

    args = parse.parse_args()
    return args

def main():
    args = parse_args()
    # Convert space-separated string to a list of floats
    values = np.array(list(map(float, args.values.split())))/args.batch_size
    mean = np.mean(values)
    std = np.std(values)

    # Determine file to write
    filename = "/gluster/home/ofrebato/baler/loss_datas/sep_loss_data.txt" if args.sep else "/gluster/home/ofrebato/baler/loss_datas/inc_loss_data.txt"
    with open(filename, "a") as f:
        f.write(f"{args.batch_size}, {mean}, {std}\n")
    print(f"Saved to {filename}")
    return 0

main()