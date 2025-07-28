import sys
import numpy as np

from numba import njit

from ase.io import read
from ase.io.cube import write_cube
from ase.io.trajectory import Trajectory

@njit
def update_voxels(voxels, hinv, rtarget, da, db, dc ):
    # Map to original unit cell
    starget = (hinv @ rtarget.T).T
    starget -= np.floor(starget)

    for s in starget:
        abin = int(np.floor(s[0] / da))
        bbin = int(np.floor(s[1] / db))
        cbin = int(np.floor(s[2] / dc))
        voxels[abin][bbin][cbin] += 1
    return voxels




crystal = read(sys.argv[1]) # ../../BEA-Sn9.xyz
traj = Trajectory(sys.argv[2]) # ../langevin.traj
target_atom = np.load(sys.argv[3])   # ../../indices/solvent_O_atoms_indices.npy
start_frame = int(sys.argv[4]) # 30000

nx = 128
ny = 128
nz = 128

voxels = np.zeros([nx, ny, nz])

da = 1.0 / nx
db = 1.0 / nx
dc = 1.0 / nx

# Map back to the original unit cell
hmat = crystal.get_cell().T
hinv = np.linalg.inv(hmat)

frame = 0
nframes = len(traj)
print("Skipping to frame", start_frame, "...")
for atoms in traj:

    if frame < start_frame:
        pass
    else:
        if frame % 1000 == 0: print(f"Processing frame {frame + 1} / {nframes}")

        rcoord = atoms.get_positions()
        #rtarget = rcoord[[atom.index for atom in atoms if atom.symbol == target_atom]]
        rtarget = rcoord[target_atom]
        voxels = update_voxels(voxels, hinv, rtarget, da, db, dc)


    frame += 1

voxels /= voxels.max()


fh = open("density.cube", "w")
write_cube(fh, crystal, data=voxels)
fh.close()
