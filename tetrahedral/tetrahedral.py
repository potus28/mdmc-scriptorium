import sys
import numpy as np
import time
from scipy.special import comb
from numba import njit
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt


traj = Trajectory(sys.argv[1])            # ../langevin.traj
center_atom_indices = np.load(sys.argv[2])   # ../../indices/zeolite_Sn_atoms_indices.npy
coordinating_atom_indices = np.load(sys.argv[3]) # ../../indices/zeolite_SnO_atoms_indices.npy


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
ncoordinating = len(coordinating_atom_indices[0])
nangles = comb(ncoordinating, 2, exact = True)



tetrahedrals = np.zeros((nframes, ncenters, nangles))

frame = 0
for atoms in traj:
    if frame % 1000 == 0: print(f"Processed frame {frame+1} / {len(traj)}")
    r = atoms.get_positions()
    hmat = atoms.get_cell().T
    for c in range(ncenters):
        center = center_atom_indices[c]
        coord = coordinating_atom_indices[c]
        tetrahedrals[frame][c] = calc_angle_combinations(r, center, coord, nangles, hmat)


    frame += 1

np.save("tetrahedral.npy", tetrahedrals)



