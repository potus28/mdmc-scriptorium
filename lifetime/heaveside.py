import sys
from numba import njit
import numpy as np
from ase.io.trajectory import Trajectory


traj = Trajectory(sys.argv[1])
groupi = np.load(sys.argv[2])
groupj = np.load(sys.argv[3])
rcut = float(sys.argv[4])

nframes = len(traj)

Harray = np.zeros((nframes, len(groupi), len(groupj)))

@njit
def get_distance_mic(ri, rj, hmat):
    hmatinv = np.linalg.inv(hmat)
    si = hmatinv @ ri
    sj = hmatinv @ rj
    ds = si - sj
    ds -= np.rint(ds)
    dr = hmat @ ds
    drmag = np.linalg.norm(dr)

    return drmag



frame = 0
for atoms in traj:
    if frame % 100 == 0: print(f"Processing frame {frame+1} / {nframes}")
    r = atoms.get_positions()
    hmat = atoms.get_cell().T
    for i, i_atomidx in enumerate(groupi):
        for j, j_atomidx in enumerate(groupj):
            ri = r[i_atomidx].T
            rj = r[j_atomidx].T
            rij = get_distance_mic(ri, rj, hmat)
            if rij <= rcut: Harray[frame][i][j] = 1
    frame += 1


np.save("Harray.npy", Harray)
