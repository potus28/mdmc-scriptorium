import sys
from ase.io import read, write
from gpaw import GPAW, PW

atoms = read(sys.argv[1])

k = 8
atoms.calc = GPAW(
        xc="PBE",
        mode=PW(500),
        kpts=(k, k, k)
        )

atoms.get_potential_energy()

atoms.calc.write('calc.gpw')


