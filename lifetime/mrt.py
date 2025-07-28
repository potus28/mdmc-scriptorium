import sys
import numpy as np
from numba import njit
from ase.io import read, write, iread
from ase.io.trajectory import Trajectory

timestep = float(sys.argv[1])
ncorr = int(sys.argv[2])
start = int(sys.argv[3])

Harray = np.load("Harray.npy")[start:]

@njit
def calc_restime(Harray, ncorr):
    restime, norm = np.zeros(ncorr), np.zeros(ncorr)

    n = 0
    nframes = len(Harray)
    for H in Harray: # loop over time origins
        if n % 1000 == 0: print(f"Processed frame {n+1} / {nframes}")

        Nref = len(H[0]) # number of reference points
        for j in range(Nref):
            Hj = H[:,j]
            normalize_factor = np.sum(Hj) # Gives the number of atoms initially near j
            #normalize_factor *= Nref # Number of j's
            # Number of j's gets taken into account from calculating the norm

            if normalize_factor != 0:
                maxn = min(nframes, n + ncorr) - n # Keeps us from indexing out of bounds
                for t in range(maxn):
                    Hn = Harray[n + t]
                    Hnj = Hn[:,j]
                    restime[t] += np.dot(Hnj, Hj) / (normalize_factor)
                    norm[t] += 1
        n += 1

    restime /= norm
    return restime

restime = calc_restime(Harray, ncorr)
tau = timestep * np.linspace(0, ncorr - 1, ncorr)
np.save("tau.npy", tau)
np.save("restime.npy", restime)

