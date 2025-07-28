import numpy as np
import matplotlib.pyplot as plt


x = np.load("tau.npy")
y = np.load("msd.npy")

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(3.5, 3.5), dpi=450, sharex = True, constrained_layout = True)

axs.loglog()

axs.plot(x, y)

exact = x*6
# plot the exact result
axs.plot(x, exact, color="black", linestyle="--", label=r'$y=2 D\tau$')


axs.set_ylabel(r"MSD($\tau$) (Å)")

axs.set_xlabel(r"$\tau$ (ps)")


plt.show()

