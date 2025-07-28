import sys
import numpy as np
from ase import units, build
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.eos import EquationOfState
from gpaw import GPAW, PW



atoms = read(sys.argv[1])

k = 8
atoms.calc = GPAW(
        xc="PBE",
        mode=PW(500),
        kpts=(k, k, k)
        )


v_o = atoms.get_volume()
cell = atoms.get_cell()
traj = Trajectory('eos.traj', 'w')

for x in np.linspace(0.95, 1.05, 5):
    atoms.set_cell(cell * x, scale_atoms=True)
    atoms.get_potential_energy()
    traj.write(atoms)


configs = Trajectory("eos.traj", "r")

volumes = [atoms.get_volume() for atoms in configs]
energies = [atoms.get_potential_energy() for atoms in configs]
eos = EquationOfState(volumes, energies)

v0, e0, B = eos.fit()
#print(B / units.kJ * 1.0e24, 'GPa')
#eos.plot('eos.png')

s = (v0 / v_o)**(1./3.)

atoms.set_cell(s*cell, scale_atoms = True)
write("optimized.xyz", atoms)

