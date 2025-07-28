import sys
import time
import json

import numpy as np
from numba import njit
from scipy.special import comb
import matplotlib.pyplot as plt

from ase.io.trajectory import Trajectory
from ase.neighborlist import NeighborList, natural_cutoffs


traj = Trajectory(sys.argv[1])            # ../langevin.traj
center_atom_indices = np.load(sys.argv[2])   # ../../indices/zeolite_Sn_atoms_indices.npy
#coordinating_atom_indices = np.load(sys.argv[3]) # ../../indices/zeolite_SnO_atoms_indices.npy
coordinating_atom_symbol = sys.argv[3] # O
start_frame = int(sys.argv[4]) # 30000

kbond = float(sys.argv[5]) # 1.2

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


@njit
def isbonded(rci, rcj, dr, k = 1.2):
    return dr <= k *(rci + rcj)


@njit
def calc_angle_combinations(r, central_atom, coord_atoms, k, hmat):
    angle = np.zeros(k)
    ncoord = len(coord_atoms)
    rcenter = r[central_atom]
    k = 0


    for i in range(0, ncoord -1):
        ri = r[coord_atoms[i]]
        for j in range(i+1, ncoord):
            rj = r[coord_atoms[j]]
            rci = get_distance_vector_mic(rcenter, ri, hmat)
            rcj = get_distance_vector_mic(rcenter, rj, hmat)
            angle[k] = get_angle(rci, rcj)
            k += 1
    return angle

@njit
def get_angle(u, v):
    umag = np.linalg.norm(u)
    uhat = u / umag
    vmag = np.linalg.norm(v)
    vhat = v / vmag
    udotv = np.dot(uhat, vhat)
    angle = np.arccos(udotv)
    return angle


nframes = len(traj)
ncenters = len(center_atom_indices)

tetrahedrals = {}
for i in range(start_frame, nframes):
    tetrahedrals[str(i)] = []

coord_atom_indices = np.array(
        [atom.index for atom in traj[0] if atom.symbol == coordinating_atom_symbol]
)

frame = start_frame
print(f"Skipping to frame {start_frame}...")
for atoms in traj[start_frame:]:

    if frame % 1000 == 0: print(f"Processed frame {frame+1} / {nframes}")

    r = atoms.get_positions()
    hmat = atoms.get_cell().T
    syms = atoms.get_chemical_symbols()
    cutoffs = natural_cutoffs(atoms)

    for center in center_atom_indices:
        coord = []
        for c_idx in coord_atom_indices:
            ri = r[center]
            rj = r[c_idx]

            dr = get_distance_mic(ri, rj, hmat)

            rci = cutoffs[center]
            rcj = cutoffs[c_idx]

            if isbonded(rci, rcj, dr, k = kbond):
                coord.append(c_idx)

        ncoordinating = len(coord)
        nangles = comb(ncoordinating, 2, exact = True)

        tetrahedrals[str(frame)].append(calc_angle_combinations(r, center, coord, nangles, hmat))

    frame += 1



for key, value in tetrahedrals.items():
    tetrahedrals[key] = np.concatenate(value)
data = np.concatenate(list(tetrahedrals.values()))
np.save("tetrahedral-otf.npy", data)

