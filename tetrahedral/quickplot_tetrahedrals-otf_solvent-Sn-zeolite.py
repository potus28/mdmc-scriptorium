import numpy as np
import matplotlib.pyplot as plt



tetrahedrals = np.load("tetrahedral-otf-select.npy")

hist, bin_edges = np.histogram(tetrahedrals, bins=100, density=True)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

x = bin_centers * 180.0 / np.pi
y = hist

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(3.5, 3.5), dpi=450, sharex = True, constrained_layout = True)

axs.plot(x, y)

axs.set_ylabel("Tetrahedral Distribution Function")
axs.set_xlabel(r"$\theta$ (degrees)")

plt.show()

