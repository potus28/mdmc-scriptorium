import sys
import numpy as np
from ase.io import read, write



atoms = read(sys.argv[1])
reference_molecule = read(sys.argv[2])

natoms = len(atoms)


solvent_indices = np.arange(natoms)
nsolventmolecules = natoms // len(reference_molecule)

solvent_indices = solvent_indices.reshape((nsolventmolecules, len(reference_molecule)))

np.save("sol_group.npy", solvent_indices)



