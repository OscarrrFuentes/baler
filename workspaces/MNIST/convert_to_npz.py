import numpy as np
import struct


def load_mnist_images(file_path):
    with open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images


def load_mnist_labels(file_path):
    with open(file_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# Replace these paths with the actual paths where your extracted files are located
train_images = load_mnist_images(
    "/Users/oscarfuentes/baler_masters/baler/workspaces/MNIST/mnist_raw_data/train-images.idx3-ubyte"
)
train_labels = load_mnist_labels(
    "/Users/oscarfuentes/baler_masters/baler/workspaces/MNIST/mnist_raw_data/train-labels.idx1-ubyte"
)
test_images = load_mnist_images(
    "/Users/oscarfuentes/baler_masters/baler/workspaces/MNIST/mnist_raw_data/t10k-images.idx3-ubyte"
)
test_labels = load_mnist_labels(
    "/Users/oscarfuentes/baler_masters/baler/workspaces/MNIST/mnist_raw_data/t10k-labels.idx1-ubyte"
)

# Combine training and test data
images = np.concatenate([train_images, test_images], axis=0)
labels = np.concatenate([train_labels, test_labels], axis=0)

# Save as .npz
output_filename = (
    "/Users/oscarfuentes/baler_masters/baler/workspaces/MNIST/data/mnist_combined.npz"
)
np.savez_compressed(output_filename, data=images, names=labels)

print("MNIST data saved as mnist_combined.npz")
