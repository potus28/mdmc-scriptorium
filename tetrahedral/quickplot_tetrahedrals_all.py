import numpy as np
import matplotlib.pyplot as plt



def plot_tetrahedral(filename, ax, label):
    tetrahedrals = np.load(filename)
    hist, bin_edges = np.histogram(tetrahedrals, range = (0, np.pi), bins=100, density=True)

    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    x = bin_centers * 180.0 / np.pi
    y = hist
    ax.plot(x, y, label = label)

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(3.5, 3.5), dpi=450, sharex = True, constrained_layout = True)


plot_tetrahedral("tetrahedral-otf.npy", axs, "O-Sn-O")
plot_tetrahedral("tetrahedral.npy", axs, "OZ-Sn-OZ")
plot_tetrahedral("tetrahedral-otf-select.npy", axs, "OS-Sn-OZ")

axs.set_ylabel("Tetrahedral Distribution Function")
axs.set_xlabel(r"$\theta$ (degrees)")
axs.legend()

plt.show()

