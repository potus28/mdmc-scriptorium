import numpy as np
from ase import Atoms
from ase.io import read
from ase.visualize import view
from define_pores import CylindricalPore

atoms = read("/Volumes/Woody_Seagate_Desktop_Drive/Woody/Projects/DOE/mlip-zeolites/md/sn-bea/reference_structures/zeolites/super_cells/st9.xyz")


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

atoms_100 = Atoms(
        "Ar"*10,
        pore_100_lower_left.centerline
        )

pore_100_lower_right = CylindricalPore(
        center = np.array([0.5*a, 14.758, 0.05113]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 10
        )

atoms_100 += Atoms(
        "Li"*10,
        pore_100_lower_right.centerline
        )

pore_100_upper_left = CylindricalPore(
        center = np.array([0.5*a, 10.56475, 13.15175]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 10
        )

atoms_100 += Atoms(
        "Kr"*10,
        pore_100_upper_left.centerline
        )

pore_100_upper_right = CylindricalPore(
        center = np.array([0.5*a, 23.22614, 13.15175]),
        radius = 4.0,
        length = a,
        direction = np.array([1.0, 0., 0.0]),
        ncenters = 10
        )

atoms_100 += Atoms(
        "K"*10,
        pore_100_upper_right.centerline
        )


# Looking down the 010 vector with z pointing right and x pointing up

pore_010_lower_left = CylindricalPore(
        center = np.array([10.56475, 0.5*b, 6.653]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 10
        )

atoms_010 = Atoms(
        "Cs"*10,
        pore_010_lower_left.centerline
        )


pore_010_upper_left = CylindricalPore(
        center = np.array([23.22614, 0.5*b, 6.653]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 10
        )

atoms_010 += Atoms(
        "Pb"*10,
        pore_010_upper_left.centerline
        )

pore_010_lower_right = CylindricalPore(
        center = np.array([2.09661, 0.5*b, 19.75325]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 10
)


atoms_010 += Atoms(
        "Fe"*10,
        pore_010_lower_right.centerline
        )


pore_010_upper_right = CylindricalPore(
        center = np.array([14.758, 0.5*b, 19.75325]),
        radius = 4.0,
        length = b,
        direction = np.array([0.0, 1.0, 0.0]),
        ncenters = 10
        )

atoms_010 += Atoms(
        "Pt"*10,
        pore_010_upper_right.centerline
        )


view(atoms + atoms_100 + atoms_010)

