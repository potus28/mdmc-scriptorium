import sys
import pandas as pd
import numpy as np

from ase import units
from ase.io import iread, read, write
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator

def read_cassandra_prp(filename):
    with open(filename) as fh:
        #First line
        fh.readline()
        #Secondline
        column_names = fh.readline().split()
        del column_names[0]

    return pd.read_table(filename, skiprows=3, names=column_names, delim_whitespace=True)



prefix = sys.argv[1]
hfile = f"{prefix}.H"
prpfile = f"{prefix}.prp"
xyzfile = f"{prefix}.xyz"

df = read_cassandra_prp(prpfile)

for column in df.columns:
    arr = np.array(df[column])
    np.save(f"{prpfile}.{column}.npy", arr)



'''
prp = np.loadtxt(prpfile, skiprows=3)
traj = iread(xyzfile, index=":")

energies = prp[:,1] * units.kJ / units.mol

cells = []
with open(hfile, "r") as fh:
    linetype = "volume"
    for line in fh.readlines():
        s = line.split()
        if linetype == "volume":
            volume = float(s[0])
            linetype = "x"

        elif linetype == "x":
            ax = float(s[0])
            bx = float(s[1])
            cx = float(s[2])
            linetype = "y"

        elif linetype == "y":
            ay = float(s[0])
            by = float(s[1])
            cy = float(s[2])
            linetype = "z"

        elif linetype == "z":
            az = float(s[0])
            bz = float(s[1])
            cz = float(s[2])
            linetype = "blank"

        elif linetype == "blank":
            linetype = "nspecies"

        elif linetype == "nspecies":
            nspecies = int(s[0])
            linetype = "nmolecules"

        elif linetype == "nmolecules":
            molid = int(s[0])
            nmols = int(s[1])

            if molid == nspecies:
                cell = [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]
                cells.append(cell)
                linetype = "volume"


outtraj = Trajectory(f"{prefix}.traj", 'w')

iframe = 0
nframes = len(energies)
for atoms in traj:

    en = energies[iframe]
    cell = cells[iframe]
    del atoms[[atom.symbol == 'Os' for atom in atoms]]
    atoms.set_cell(cell)
    atoms.set_pbc(True)
    calc = SinglePointCalculator(atoms, energy = en)
    atoms.calc = calc
    outtraj.write(atoms)

    print(f"Processed frame {iframe} / {nframes}")
    iframe += 1
'''

