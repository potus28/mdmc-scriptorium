import numpy as np
import matplotlib.pyplot as plt



psd = 2.0*np.load('pore_radii.npy')
hist, bin_edges = np.histogram(psd, bins = 15, range=(1.5, 7.0), density=True)

bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

plt.plot(bin_centers, hist)

plt.show()
