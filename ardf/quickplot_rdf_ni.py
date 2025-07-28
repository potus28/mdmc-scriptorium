import numpy as np
import matplotlib.pyplot as plt

x = np.load("x_centers.npy")
y1 = np.load("rdf.npy")
y2 = np.load("number_integral.npy")


fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize=(3.5, 3.5), dpi=450, sharex = True, constrained_layout = True)

axs[0].plot(x, y1)
axs[1].plot(x, y2)

axs[0].set_ylabel("RDF")
axs[1].set_ylabel("NI")

axs[1].set_xlabel("$r$ (Å)")

plt.show()

