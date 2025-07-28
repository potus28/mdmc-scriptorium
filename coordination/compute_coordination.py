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

# Compute coordination number of group_i to group_j
group_i = np.load(sys.argv[2])
group_j = np.load(sys.argv[3])
prefix = sys.argv[4]

n_group_i = len(group_i)
nframes = len(traj)
coordination_ij = np.zeros((nframes, n_group_i), dtype=int)

@njit
def compute_cn(covalent_radii, s, h, group_i, group_j, k):
    cn = np.zeros(len(group_i))
    ii = 0
    for atomi in group_i:
        si = s[atomi]
        cri = covalent_radii[atomi]

        for atomj in group_j:
            sj = s[atomj]
            crj = covalent_radii[atomj]
            ds = si - sj
            ds -= np.rint(ds)
            dr = h @ ds.T

            dist = np.linalg.norm(dr)
            if dist <= k * (cri + crj) and atomi != atomj:
                #print(atomi, atomj, dist)
                cn[ii] += 1

        ii += 1
    return cn

frame = 0
for atoms in traj:
    if frame % 1000 == 0:
        print(f"Processing frame {frame + 1} / {nframes}")
    covalent_radii = np.array(natural_cutoffs(atoms))
    s = atoms.get_scaled_positions()
    cell = atoms.get_cell()
    h = cell.T
    coordination_ij[frame] = compute_cn(covalent_radii, s, h, group_i, group_j, k =1.6)
    frame += 1

np.save(f"{prefix}.coordination.npy", coordination_ij)

