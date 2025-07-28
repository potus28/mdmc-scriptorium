import sys
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory


traj = Trajectory(sys.argv[1])
xbins = np.load(sys.argv[2])
rdf = np.load(sys.argv[3])
observed_atom_indices = np.load(sys.argv[4])

N = len(observed_atom_indices)
V = traj[0].get_volume()

integrand = rdf * xbins * xbins

number_integral = 4.0 * np.pi * N / V * cumtrapz(integrand, xbins, initial = 0)

np.save("number_integral.npy", number_integral)



