import sys

from numba import njit
import numpy as np


from ase.io import read, write, iread

from ase.build import molecule
from ase.neighborlist import get_connectivity_matrix
from ase.neighborlist import natural_cutoffs
from ase.neighborlist import NeighborList

cns = np.load(sys.argv[1])
prefix = sys.argv[2]
upper_half = bool(int(sys.argv[3]))
szs = np.load(sys.argv[4])

@njit
def compute_cn(szs, cns, upper_half):

    nframes = len(cns)
    nrxns = np.zeros((nframes, 2))

    for frame in range(nframes):
        if frame > 0:
            if frame % 1000 == 0:
                print(f"Calculating reactions on frame {frame} / {nframes}")

            sz = szs[frame]
            f2 = cns[frame]
            f1 = cns[frame - 1]
            df = f2 - f1


            # Dissocation: 1 -> 0 ( df = 0 - 1 = -1 )


            if upper_half:
                ndis = np.sum((df == -1) & (sz > 0.5))
            else:
                ndis = np.sum((df == -1) & (sz < 0.5))


            if ndis > 0:
                nrxns[frame][0] = ndis + nrxns[frame-1][0]
            else:
                nrxns[frame][0] = nrxns[frame-1][0]

            # Recombination: 0 -> 1 (df = 1 - 0 = 1)
            if upper_half:
                ncom = np.sum((df == 1) & (sz > 0.5))
            else:
                ncom = np.sum((df == 1) & (sz < 0.5))

            if ncom > 0:
                nrxns[frame][1] = ncom + nrxns[frame-1][1]
            else:
                nrxns[frame][1] = nrxns[frame-1][1]


    return nrxns


# Have to divide by two to prevent overcounting
nrxns = compute_cn(szs, cns, upper_half) / 2.0
np.save(f"{prefix}.{upper_half}.dis.npy", nrxns[:,0])
np.save(f"{prefix}.{upper_half}.com.npy", nrxns[:,1])

