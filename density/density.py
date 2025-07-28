import sys
import numpy as np
from ase import units
from ase.io.trajectory import Trajectory

traj = Trajectory(sys.argv[1])
nframes = len(traj)
density = np.zeros(nframes)

frame = 0

g_to_kg = 1.0E-3
angstrom3_to_m3 = 1.0E-30

for atoms in traj:
    if frame % 1000 == 0: print(f"Processing frame {frame} / {nframes}...")
    m = np.sum(atoms.get_masses()) * g_to_kg / units.mol  # kg
    v = atoms.get_volume() * angstrom3_to_m3 # m^3
    density[frame] = m/v
    frame +=1

np.save("density.npy", density)
