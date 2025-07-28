import sys
import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory


traj = Trajectory(sys.argv[1])
#atoms_ref = read(sys.argv[2])
atoms_ref = traj[0]
cell_ref = atoms_ref.get_cell_lengths_and_angles()


nframes = len(traj)
percent_erros = np.zeros((nframes, 6))

def get_percent_error(actual, expected):
    return 100.0 * (actual - expected) / expected

frame = 0
for atoms in traj:
    if (frame + 1) % 1000 == 0: print(f"Processing frame {frame+1} / {nframes} ...")
    cell = atoms.get_cell_lengths_and_angles()
    percent_erros[frame] = get_percent_error(cell, cell_ref)
    frame += 1

print(percent_erros)

np.save("a_error.npy", percent_erros[:,0])
np.save("b_error.npy", percent_erros[:,1])
np.save("c_error.npy", percent_erros[:,2])
np.save("alpha_error.npy", percent_erros[:,3])
np.save("beta_error.npy", percent_erros[:,4])
np.save("gamma_error.npy", percent_erros[:,5])

