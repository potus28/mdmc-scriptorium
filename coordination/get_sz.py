import sys

from numba import njit
import numpy as np


from ase.io import read, write, iread
from ase.io import Trajectory

from ase.build import molecule
from ase.neighborlist import get_connectivity_matrix
from ase.neighborlist import natural_cutoffs
from ase.neighborlist import NeighborList

traj = Trajectory(sys.argv[1])
group_i = np.load(sys.argv[2])
prefix = sys.argv[3]

nframes = len(traj)
sz = np.zeros((nframes, len(group_i)))

frame = 0
for atoms in traj:
    if frame % 1000 == 0:
        print(f"Reading coordinates from frame {frame} / {nframes}")
    atoms.wrap()
    sz[frame] = atoms.get_scaled_positions()[group_i][:,2]
    frame += 1


np.save(f"{prefix}.sz.npy", sz)

