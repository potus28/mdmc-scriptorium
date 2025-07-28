import numpy as np
from ase.io.trajectory import Trajectory



start = 0
traj = Trajectory("../gcmc.traj")[start:]
target_element = "N"
number_of_sites = 32 #atop Pt top and bottom


dz = 2.5


top_indices = np.arange(80, 96, dtype = int)
bottom_indices = np.arange(0, 16, dtype = int)


coverage = np.zeros(len(traj))
frame =0
for atoms in traj:

    atoms.wrap()
    r = atoms.get_positions()
    rz = atoms.get_positions()[:,2]

    rzi = rz[[atom.index for atom in atoms if atom.symbol == target_element]]
    
    # top
    rzj = rz[top_indices]
    rzj_avg = np.mean(rzj)
    invicinity_top = (rzj_avg - dz < rzi) & (rzi < rzj_avg + dz)
    # bottom
    rzj = rz[bottom_indices]
    rzj_avg = np.mean(rzj)
    invicinity_bottom = (rzj_avg - dz < rzi) & (rzi < rzj_avg + dz)
    
    # total
    nvicinity = np.sum(invicinity_top) + np.sum(invicinity_bottom)

    # coverage
    coverage[frame] = nvicinity / number_of_sites * 100.0
    frame += 1


np.save("coverage.npy", coverage)

