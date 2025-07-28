import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig, axs = plt.subplots(
        nrows=1, ncols=1 ,
        figsize =[3.5, 3.5], dpi = 450,
        sharex=False, sharey=False,
        layout="constrained"
        )

def plot_heatmap(fig, ax, x, y, z, make_cbar = True):
    im = ax.pcolormesh(x, y, z.T, cmap = "CMRmap", shading='gouraud', antialiased=True)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    if make_cbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"Occurence")


x = np.load("x_centers.npy")
y = np.load("theta_centers.npy") * 180.0/np.pi
z = np.load("occurences.npy")

plot_heatmap(fig, axs, x, y, z)

axs.set_ylabel(r"$\theta$ (degrees)")
axs.set_xlabel("$r$ (Å)")

plt.show()
