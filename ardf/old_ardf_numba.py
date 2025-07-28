import sys
import numpy as np
from numba import njit
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt



def main():
    maxr = 10.0
    maxtheta = np.pi
    nbins = 100
    start_frame = 30000
    logfreq = 1000

    traj = Trajectory("../langevin.traj")
    ref_atom_indices = np.load("../../indices/zeolite_Sn_atoms_indices.npy")
    obs_atom_indices = np.load("../../indices/solvent_O_atoms_indices.npy")

    ref_atom_vectors_fname = np.load("r_Sn-OS.npy")
    obs_atom_vectors_fname = np.load("r_OM.npy")

    #traj = Trajectory(sys.argv[1])

    #ref_atom_indices = np.load(sys.argv[2])
    #ref_atom_vectors_fname = sys.argv[3]

    #obs_atom_indices = np.load(sys.argv[4])
    #obs_atom_vectors_fname = sys.argv[5]



    samevectors = (ref_atom_vectors_fname == obs_atom_vectors_fname)
    ref_atom_vectors = np.load(ref_atom_vectors_fname)
    obs_atom_vectors = np.load(obs_atom_vectors_fname)


    dr = maxr / nbins
    dtheta = maxtheta / nbins
    r_histo = np.zeros(nbins)
    theta_histo = np.zeros(nbins)
    histo_2d = np.zeros((nbins, nbins))

    frame = 0
    Nframes = 0
    print(f"Skipping to frame {start_frame}...")
    for atoms in traj:

        if frame >= start_frame:

            Nframes += 1
            if frame % logfreq == 0: print(f"Processed frame {frame+1} / {len(traj)}")

            ref_vectors = ref_atom_vectors[frame]
            obs_vectors = obs_atom_vectors[frame]

            for i, ref in enumerate(ref_atom_indices):
                # Only count if ref is not in obs_atom
                isnotequal = (ref != obs_atom_indices)
                distances = atoms.get_distances(ref, obs_atom_indices[isnotequal], mic = True)

                if samevectors:
                    vec_indices = np.arange(len(ref_vectors))
                    isnotequal = (i != vec_indices)
                    angles = get_angles(ref_vectors[i], obs_vectors[isnotequal])
                else:
                    angles = get_angles(ref_vectors[i], obs_vectors)

                r_histo = accumulate_histogram(distances, r_histo, dr, maxr)
                theta_histo = accumulate_histogram(angles, theta_histo, dtheta, maxtheta)
                histo_2d = accumulate_histogram_2d(histo_2d, distances, dr, maxr, angles, dtheta, maxtheta)

        else:
            pass

        frame += 1

    Nobs = len(obs_atom_indices)
    Nref = len(ref_atom_indices)
    V = traj[0].get_volume()

    save_edges_and_centers(maxr, dr, "x")
    save_edges_and_centers(maxtheta, dtheta, "theta")

    np.save("adf.npy", theta_histo)
    save_rdf(r_histo, nbins, dr, Nref, Nobs, V, Nframes)
    save_ardf(histo_2d, nbins, dr, dtheta, Nref, Nobs, V, Nframes)


def save_edges_and_centers(xmax, dx, label):
    edges = np.arange(0, xmax + dx, dx)
    centers = (edges[:-1] + edges[1:]) * 0.5
    np.save(f"{label}_centers.npy", centers)
    np.save(f"{label}_edges.npy", edges)

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
def get_angles(u, varray):
    i = 0
    angles = np.zeros(len(varray))
    umag = np.linalg.norm(u)
    uhat = u / umag
    for v in varray:
        vmag = np.linalg.norm(v)
        vhat = v / vmag
        udotv = np.dot(uhat, vhat)
        angles[i] = np.arccos(udotv)
        i += 1
    return angles

@njit
def accumulate_histogram_2d(histo, xdat, dx, xmax, ydat, dy, ymax):

    for i in range(len(xdat)):
        x = xdat[i]
        y = ydat[i]
        if x <= xmax and y <= ymax:
            xbin = int(np.floor(x / dx))
            ybin = int(np.floor(y / dy))
            histo[xbin][ybin] += 1
    return histo

@njit
def accumulate_histogram(dat, histo, dx, xmax):
    for x in dat:
        if x <= xmax:
            xbin = int(np.floor(x / dx))
            histo[xbin] += 1
    return histo


def save_ardf(histo, nbins, delx, deltheta, Nref, Nobs, V, Nframes):
    np.save("occurences.npy", histo)

    ardf = np.zeros((nbins, nbins))
    const = (2.0 / 3.0) * np.pi * Nobs * Nref * Nframes / V

    x = 0
    for i in range(nbins):
        volx = (x + delx)**3 - (x)**3
        theta = 0
        for j in range(nbins):
            voltheta = np.cos(theta) - np.cos(theta + deltheta)
            vol = volx * voltheta
            ardf[i][j] = histo[i][j] / vol / const

            theta += deltheta
        x += delx


    np.save("ardf.npy", ardf)


def save_rdf(histo, nbins, delx, Nref, Nobs, V, Nframes):
    rdf = np.zeros(nbins)
    const = (4.0 / 3.0) * np.pi * Nobs * Nref * Nframes / V
    x = 0
    for i in range(nbins):
        vol = (x + delx)**3 - (x)**3
        rdf[i] = histo[i] / vol / const
        x += delx

    np.save("rdf.npy", rdf)



if __name__ == '__main__':
    main()

