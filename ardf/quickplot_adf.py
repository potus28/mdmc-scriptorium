import numpy as np
import matplotlib.pyplot as plt

x = np.load("theta_centers.npy") * 180.0 / np.pi
y = np.load("adf.npy")

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(3.5, 3.5), dpi=450, sharex = True, constrained_layout = True)

axs.plot(x, y)

axs.set_ylabel("ADF")
axs.set_xlabel(r"$\theta$ (degrees)")

plt.show()

