import numpy as np

data = np.load("/gluster/home/ofrebato/baler/workspaces/MNIST/data/mnist_combined.npz")
np.savez("/gluster/home/ofrebato/baler/workspaces/MNIST/data/mnist_combined.npz", data=data["data"], names = data["names"])
