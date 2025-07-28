import numpy as np
from numba import njit




@njit
def get_angles(u, varray):
    i = 0
    angles = np.zeros(len(varray))
    umag = np.linalg.norm(u)
    uhat = u/umag
    for v in varray:
        vmag = np.linalg.norm(v)
        vhat = v/vmag
        udotv = np.dot(uhat, vhat)
        angles[i] = np.arccos(udotv)
        i += 1
    return angles



rOH = np.load("rOH.npy")
angles = get_angles(rOH, rOH)

nbins = 100

maxtheta = np.pi
dtheta = maxtheta / nbins

histo = np.zeros(nbins)


for frame in rOH:
    for i, vec in enumerate(frame):
        ref_vec = frame[i]:
            for j, vec in enumerate(frame):


