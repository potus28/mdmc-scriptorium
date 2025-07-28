import numpy as np
import matplotlib.pyplot as plt


x = np.load("tau.npy")
y = np.load("tdm_acf.npy")

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(3.5, 3.5), dpi=450, sharex = True, constrained_layout = True)

axs.plot(x, y)

axs.set_ylabel(r"$\Phi$($\tau$)")
axs.set_xlabel(r"$\tau$ (ps)")

plt.show()

