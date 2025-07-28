
import sys
import numpy as np
from numba import njit
from ase.io.trajectory import Trajectory
from ase.neighborlist import NeighborList, natural_cutoffs, get_connectivity_matrix

@njit
def get_distance_mic(ri, rj, hmat):
    dr = get_distance_vector_mic(ri, rj, hmat)
    drmag = np.linalg.norm(dr)
    return drmag

@njit
def get_distance_vector_mic(ri, rj, hmat):
    """From vector FROM i TO j"""
    hmatinv = np.linalg.inv(hmat)
    si = hmatinv @ ri
    sj = hmatinv @ rj
    ds = sj - si
    ds -= np.rint(ds)
    dr = hmat @ ds
    return dr



traj = Trajectory('../langevin.traj')


groupi = np.load('../../indices/solvent_O_atoms_indices.npy')

prefix = "rOM"

nframes = len(traj)
vectors = np.zeros((nframes, len(groupi), 3))

frame = 0

for atoms in traj:
    if frame %100 == 0: print(f'Processing frame {frame} / {nframes}')

    r = atoms.get_positions()
    hmat = atoms.get_cell().T
    i = 0
    for atomi in groupi:
        rO = r[atomi]
        rC = r[atomi-1]
        rH = r[atomi+2]


        drOH = get_distance_vector_mic(rO, rH, hmat)
        drOC = get_distance_vector_mic(rO, rC, hmat)

        drOM = drOH / np.linalg.norm(drOH) + drOC / np.linalg.norm(drOC)
        drOM /= np.linalg.norm(drOM)

        vectors[frame][i] = drOM

        i += 1

    frame += 1


np.save(f'{prefix}.npy', vectors)
