import sys
import numpy as np
from scipy.integrate import cumtrapz
from numba import njit
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt

@njit
def accumulate_histogram(distances, histo, delx):
    for i in range(1, distances.shape[0]):
        for j in range(0, i):
            if distances[i,j] > maxd : continue
            xbin = int(np.floor(distances[i,j] / delx))
            histo[xbin] += 2
    return histo

@njit
def accumulate_histogram_1d(distances, histo, delx):
    for d in distances:
        if d > maxd : continue
        xbin = int(np.floor(d / delx))
        histo[xbin] += 1
    return histo

def normalize_rdf(histo, nbins, delx, Nref, Nobserved, V, nf):
    xbins = np.zeros(nbins)
    const = (4.0 / 3.0) * np.pi *(Nobserved/V)
    for i in range(nbins):
        if i == 0:
            xbins[i] = 0.5*delx
            vol = const * delx**3
        else:
            xbins[i] = (i+0.5)*delx
            vol = const * ((delx*(i))**3 - (delx*(i-1))**3 )

        histo[i] /= (vol*nf*Nref)

    return xbins, histo

N = 0
V = 0.0

ref_atom_indices = np.load(sys.argv[2])
observed_atom_indices = np.load(sys.argv[3])
maxd = float(sys.argv[4])
nbins = int(sys.argv[5])
start_frame = int(sys.argv[6])

delx = maxd / nbins
histo = np.zeros(nbins)

traj = Trajectory(sys.argv[1])[start_frame:]

Nobserved = len(observed_atom_indices)
Nref = len(ref_atom_indices)
V = traj[0].get_volume()
nframes = len(traj)


frame = 0
framecount = 0
for atoms in traj:
    if frame % 1000 == 0: print(f"Processed frame {frame+1} / {len(traj)}")
    #N = len(atoms)
    #V = atoms.get_volume()
    #distances = atoms.get_all_distances(mic = True)
    if frame >= start_frame:
        for ref in ref_atom_indices:
            distances = atoms.get_distances(ref, observed_atom_indices, mic = True)
            histo = accumulate_histogram_1d(distances, histo, delx)
        framecount += 1
    frame += 1


xbins, rdf = normalize_rdf(histo, nbins, delx, Nref, Nobserved, V, framecount)

integrand = rdf * xbins * xbins
number_integral = 4.0 * np.pi * Nobserved / V * cumtrapz(integrand, xbins, initial = 0)

np.save("xbins.npy", xbins)
np.save("rdf.npy", rdf)
np.save("number_integral.npy", number_integral)


