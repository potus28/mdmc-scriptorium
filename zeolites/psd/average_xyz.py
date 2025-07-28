import sys
import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory


reference_atoms = read(sys.argv[1])
traj = Trajectory(sys.argv[2])
nx = 2
ny = 2
nz = 1


nframes = len(traj)
natoms_supercell = len(traj[0])
natoms_reference = len(reference_atoms)
nunits = natoms_supercell // natoms_reference

r_array = np.zeros((nframes, natoms_supercell, 3))
cell_array = np.zeros((nframes, 3, 3))

unit_cell_tags = np.zeros(natoms_supercell, dtype=int)

uc_count = 0
for i in range(natoms_supercell):
    unit_cell_tags[i] = uc_count
    if (i+1) % natoms_reference == 0:
        uc_count += 1

frame = 0
for atoms in traj:
    if frame % 1000 == 0: print(f"Processing frame {frame} / {nframes}")
    cell = atoms.get_cell()[:]
    cell[0] /= nx
    cell[1] /= ny
    cell[2] /= nz
    cell_array[frame] = cell

    r = atoms.get_positions()
    r_array[frame] = r

    frame += 1


average_cell = np.mean(cell_array, axis = 0)
print("Average cell", average_cell)
hmat = average_cell.T
hinv = np.linalg.inv(hmat)

average_coordinates_supercell = np.mean(r_array, axis = 0)
average_coordinates_reference = np.zeros((natoms_reference, 3))






for idx in range(natoms_reference):
    r = np.zeros(3)
    for j in range(nunits):
        if j == 0:
            r += average_coordinates_supercell[idx + natoms_reference*j]
        elif j == 1:
            r += average_coordinates_supercell[idx + natoms_reference*j] - average_cell[0]
        elif j == 2:
            r += average_coordinates_supercell[idx + natoms_reference*j] - average_cell[1]
        elif j == 3:
            r += average_coordinates_supercell[idx + natoms_reference*j] - average_cell[0] - average_cell[1]

    r /= nunits
    average_coordinates_reference[idx] = r


reference_atoms.set_cell(average_cell)
reference_atoms.set_positions(average_coordinates_reference)
write("average_coord.xyz", reference_atoms)


