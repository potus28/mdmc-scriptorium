import numpy as np
from ase.io import iread


energy_correlation = np.load('energy_correlation.npy')
traj = iread('../test.xyz')

frame = 0
for atoms in traj:
    energy_correlation[frame] = energy_correlation[frame] / len(atoms)
    frame += 1



np.save('energy_correlation_peratom.npy', energy_correlation)
