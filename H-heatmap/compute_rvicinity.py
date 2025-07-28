import sys

import numpy as np

from ase.io import read, write, iread
from ase.io import Trajectory

from ase.build import molecule
from ase.neighborlist import get_connectivity_matrix
from ase.neighborlist import natural_cutoffs
from ase.neighborlist import NeighborList

traj = Trajectory(sys.argv[1])
nframes = len(traj)

# Compute coordinates of group_i in the vicinity of the avrerage if group_j's z coordinate
group_i = np.load(sys.argv[2])
group_j = np.load(sys.argv[3])
dz = float(sys.argv[4])
start = int(sys.argv[5])
prefix = sys.argv[6]

rs_list = []

nvicinity = np.zeros(nframes)

frame = 0
for atoms in traj:
    if frame % 1000 == 0:
        print(f"Processing frame {frame + 1} / {nframes}")
    atoms.wrap()
    r = atoms.get_positions()
    rz = atoms.get_positions()[:,2]
    ri = r[group_i]
    rzi = rz[group_i]
    rzj = rz[group_j]
    rzj_avg = np.mean(rzj)

    invicinity = (rzj_avg - dz < rzi) & (rzi < rzj_avg + dz)
    r_vicinity = ri[invicinity]



    nvicinity[frame] = np.sum(invicinity)

    if frame >= start:
        rs_list.append(r_vicinity)

    frame += 1

# We have a list of arrays of length nframes with different natoms_per_frame x 3 dimennsions
# Concatenate into 1 master array and save
rs = np.concatenate(rs_list, axis = 0)

np.save(f'{prefix}.r.npy',rs)
np.save(f'{prefix}.nvicinity.npy', nvicinity)

