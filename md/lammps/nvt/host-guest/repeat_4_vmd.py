
import sys
from ase.io import read, write

atoms = read(sys.argv[1])

hmat = atoms.get_cell().T
atoms = atoms.repeat((3, 2, 2))

for atom in atoms:
    atom.position = atom.position - hmat[:,0] - hmat[:,1] - hmat[:,2]

write("VMD-Beta_A.xyz", atoms)

