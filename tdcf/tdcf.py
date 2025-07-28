
import sys
import numpy as np
import matplotlib.pyplot as plt

from numba import njit

from ase.io.trajectory import Trajectory
from ase.neighborlist import NeighborList, natural_cutoffs, get_connectivity_matrix

@njit
def calculate_dipole_moment(positions, charges):
    '''
    positions: Natoms x 3 array
    charges: 1 x Natoms array
    '''
    return np.sum((charges * positions.T).T, axis = 0)

@njit
def get_total_dipole_moment(mus):
    return np.sum(mus, axis = 0)


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
def get_unit_vector(v):
    return v / np.linalg.norm(v)


@njit
def get_bisector_vector(u, v, mirror_vector = False, unit_vector = False):
    ''' calculate the bisector vector formed between vectors u and v
        using the vector identity ||u||v + ||v||u
    '''
    b = np.linalg.norm(u)* v + np.linalg.norm(v) * u
    if mirror_vector:
        b *= -1.0
    if unit_vector:
        b = get_unit_vector(b)

    return b


# 1) Get the time series of total dipole moments
traj = Trajectory(sys.argv[1])
mol_groups = np.load(sys.argv[2])
timestep = float(sys.argv[3])
ncorr = int(sys.argv[4])
start_frame = int(sys.argv[5])

nmolecules = len(mol_groups)
nframes = len(traj)

tdm = np.zeros((nframes, 3))

frame = 0
for atoms in traj:
    if (frame + 1) % 1000 == 0 : print(f"Processing frame {frame+1} / {nframes}...")

    r = atoms.get_positions()
    #q = atoms.get_charges()
    mus = np.zeros((nmolecules, 3))
    hmat = atoms.get_cell().T

    i = 0
    for mol_indices in mol_groups:

        rC = r[mol_indices][0]
        rO = r[mol_indices][1]
        rH = r[mol_indices][3]

        rOC = get_distance_vector_mic(rO, rC, hmat)
        rOH = get_distance_vector_mic(rO, rH, hmat)

        #mu = calculate_dipole_moment(r[mol_indices], q[mol_indices])
        mu = get_bisector_vector(rOC, rOH, mirror_vector = False, unit_vector = True)
        mus[i] = mu
        i += 1

    tdm[frame] = get_total_dipole_moment(mus)

    frame += 1



@njit
def tdm_auto_correlation(array, ncorr):
    acf = np.zeros(ncorr)
    norm = np.zeros(ncorr)
    nframes = len(array)

    #norm_squared = np.linalg.norm(array, axis = 1)**2

    n = 0
    for x in array: # Loop over time averages
        if (n + 1) % 1000 == 0: print(f"Processed {n+1} / {nframes}")
        maxn = min(nframes, n + ncorr) - n # Prevents from indexing out of bounds when at the end of the array
        x0 = x # x(t)
        denominator = np.linalg.norm(x0)**2 # ||x||^2


        for i in range(maxn):
            xn = array[n + i] # x(t + i * dt)
            acf[i] += np.dot(xn, x0) / denominator
            norm[i] += 1

        n += 1

    acf /= norm # Dividing here takes the average value of all observations

    return acf




analysis_frames = tdm[start_frame:]
acf = tdm_auto_correlation(analysis_frames, ncorr)

# Convert correlation depth to time
tau = timestep * np.linspace(0, ncorr - 1, ncorr)
np.save("tau.npy", tau)
np.save("tdm_acf.npy", acf)








