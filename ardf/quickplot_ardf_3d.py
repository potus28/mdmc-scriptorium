import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



fig, ax = plt.subplots(
        nrows=1, ncols=1 ,
        figsize =[3.5, 3.5], dpi = 450,
        sharex=False, sharey=False,
        layout="constrained",
        subplot_kw={"projection": "3d"}
        )


x = np.load("x_centers.npy")
y = np.load("theta_centers.npy") * 180.0/np.pi
z = np.load("ardf.npy")





# Plot the surface.

X, Y = np.meshgrid(x, y)

im = ax.plot_surface(X, Y, z, cmap=mpl.cm.coolwarm, antialiased=False)


cbar = fig.colorbar(im, ax=ax, shrink=0.5, aspect=5)
cbar.set_label(r"g($r$,$\theta$)")



ax.set_ylabel(r"$\theta$ (degrees)")
ax.set_xlabel("$r$ (Å)")

plt.show()
