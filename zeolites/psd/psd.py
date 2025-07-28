import sys
from numba import njit
import numpy as np
from ase.io import read, write
from ase.data import atomic_numbers, vdw_radii

atoms = read(sys.argv[1])
gres = int(sys.argv[2])

rprobe = 3.41

rdict = {
        "Si": 3.80414,
        "O": 3.03315,
        "Al": 3.91105,
        "Sn": 3.98232,
        "H": 2.85088,
        }

natoms = len(atoms)
hmat = atoms.get_cell().T

syms = atoms.get_chemical_symbols()
radii = np.zeros(len(syms))
for idx, sym in enumerate(syms):
    radii[idx] = rdict[sym]
    #radii[idx] = vdw_radii[atomic_numbers[sym]]

# Create a meshgrid
x = np.linspace(0, 1, gres, endpoint=False)
y = np.linspace(0, 1, gres, endpoint=False)
z = np.linspace(0, 1, gres, endpoint=False)
xx, yy, zz = np.meshgrid(x, y, z)

# Stack the coordinates to get a 3D array of points
s_grid = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
s_coords = atoms.get_scaled_positions()
r_grid = (hmat @ s_grid.T).T

@njit
def distances_pbc_array(s1, s2, hmat):
    ds = s1 - s2
    ds -= np.rint(ds)
    dr = (hmat @ ds.T).T
    distance = np.zeros(len(dr))
    for idx in range(len(dr)):
        distance[idx] = np.linalg.norm(dr[idx])
    return distance


@njit
def distances_pbc_one_point(s1, s2, hmat):
    ds = s1 - s2
    ds -= np.rint(ds)
    dr = (hmat @ ds.T).T
    distance = np.linalg.norm(dr)
    return distance





@njit
def calc_psd(s_grid, s_coords, hmat, niter = 100000):
    ngridpts = len(s_grid)
    pore_sizes = np.zeros(ngridpts)
    for gridpnt_idx, s_pnt in enumerate(s_grid):
        print(f"Grid point {gridpnt_idx} / {ngridpts}...")

        distances_grid = distances_pbc_array(s_coords, s_pnt, hmat)

        # check if the grid point is inside an atom's sphere
        distances_adjusted_grid = distances_grid - 0.5*radii
        if np.min(distances_adjusted_grid) < 0.0:
            continue

        largest_radius = -10.0
        for _ in range(niter):
            s_center = np.random.rand(3)
            distances_center = distances_pbc_array(s_coords, s_center, hmat)

            # check for overlap
            distances_center_adjusted = distances_center - 0.5*radii
            if np.min(distances_center_adjusted) < 0.0:
                continue

            # check if point is inside the sphere
            radius = np.min(distances_center_adjusted)
            distance_from_gridpnt = distances_pbc_one_point(s_center, s_pnt, hmat)

            if distance_from_gridpnt > radius:
                continue
            if radius > largest_radius:
                largest_radius = radius

        if largest_radius < 0.0:
            largest_radius = np.min(distances_adjusted_grid)

        pore_sizes[gridpnt_idx] = largest_radius

    return pore_sizes

pore_sizes = calc_psd(s_grid, s_coords, hmat)

np.save("pore_radii.npy", pore_sizes)

