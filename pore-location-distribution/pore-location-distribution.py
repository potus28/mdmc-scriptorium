import sys
import numpy as np
from ase.io.trajectory import Trajectory
from scipy import constants
from scipy.integrate import cumtrapz
from numba import njit
import matplotlib.pyplot as plt

from define_pores import CylindricalPore

observed_atoms = np.load("../../indices/solvent_O_atoms_indices.npy")
timestep = 0.004
start_frame = 25000

traj = Trajectory("../langevin.traj")[start_frame:]

natoms = len(observed_atoms)
nframes = len(traj)


# BEA zeolite lattice lengths for a 2x2x1 supercell
a = 25.32278
b = 25.32278
c = 26.40612

# Looking down the 100 vector with y pointing right and z pointing up
pore_100_lower_left = CylindricalPore(
        center = np.array([0.5*a, 2.09661, 0.05113]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 10,
        )

pore_100_lower_right = CylindricalPore(
        center = np.array([0.5*a, 14.758, 0.05113]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 10
        )

pore_100_upper_left = CylindricalPore(
        center = np.array([0.5*a, 10.56475, 13.15175]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 10
        )

pore_100_upper_right = CylindricalPore(
        center = np.array([0.5*a, 23.22614, 13.15175]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 10
        )

# Looking down the 010 vector with z pointing right and x pointing up

pore_010_lower_left = CylindricalPore(
        center = np.array([10.56475, 0.5*b, 6.653]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 10
        )

pore_010_upper_left = CylindricalPore(
        center = np.array([23.22614, 0.5*b, 6.653]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 10
        )


pore_010_lower_right = CylindricalPore(
        center = np.array([2.09661, 0.5*b, 19.75325]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 10
)



pore_010_upper_right = CylindricalPore(
        center = np.array([14.758, 0.5*b, 19.75325]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 10
        )


def run_analysis(traj, pore, outfile):
    in_pore = np.zeros((nframes, natoms), dtype = int)

    frame = 0
    for atoms in traj:
        if frame % 1000 == 0:
            print(f"Processing frame {frame + 1} / {nframes}")
        robserved = atoms.get_positions()[observed_atoms]
        hmat = atoms.get_cell().T

        i = 0
        for r in robserved:
            if pore.is_in_pore(r, hmat):
                in_pore[frame][i] = 1
            else:
                in_pore[frame][i] = 0

            i += 1

        frame += 1

    np.save(outfile, in_pore)


run_analysis(traj, pore_100_lower_left, "100_lower_left.npy")
run_analysis(traj, pore_100_upper_left, "100_upper_left.npy")
run_analysis(traj, pore_100_lower_right, "100_lower_right.npy")
run_analysis(traj, pore_100_upper_right, "100_upper_right.npy")

run_analysis(traj, pore_010_lower_left, "010_lower_left.npy")
run_analysis(traj, pore_010_upper_left, "010_upper_left.npy")
run_analysis(traj, pore_010_lower_right, "010_lower_right.npy")
run_analysis(traj, pore_010_upper_right, "010_upper_right.npy")




