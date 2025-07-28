import numpy as np

from ase import Atoms, units
from ase.calculators.lj import LennardJones
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    MaxwellBoltzmannDistribution,
)






x = np.linspace(0, 40, 4, endpoint= False)
y = np.linspace(0, 40, 4, endpoint= False)
z = np.linspace(0, 40, 4, endpoint= False)
X,Y,Z= np.meshgrid(x,y,z)

r = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T

#atoms = Atoms("C64", r, cell = [40.0, 40.0, 40.0], pbc=True)
atoms = Trajectory("../run-0/langevin.traj")[-1]

calc = LennardJones(sigma = 3.73, epsilon = 148.0*units.kB, rc = 14.0, smooth = True)
atoms.calc = calc

dyn = Langevin(
    atoms,
    0.5 * units.fs,
    temperature_K=150.0,
    friction=0.01 / units.fs,
    logfile="langevin.log",
    trajectory="langevin.traj",
    loginterval=8,
)
#MaxwellBoltzmannDistribution(atoms, temperature_K=150)
#Stationary(atoms)
#ZeroRotation(atoms)


dyn.run(2000)

