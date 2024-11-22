import numpy as np

def main():
    losses = np.load("workspaces/MNIST/MNIST_project/output/training/loss_data.npy")[0]
    print(min(losses))

main()
