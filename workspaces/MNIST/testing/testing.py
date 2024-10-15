import numpy as np

data = np.load("../../CFD_workspace/data/CFD_animation.npz")

print(data["data"][0])
