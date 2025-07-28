import sys
import numpy as np
from ase.io.trajectory import Trajectory
from ase.io import write 

traj = Trajectory(sys.argv[1])
nframes = len(traj)

k = 0
for atoms in traj:
    if k % 1000 == 0: print(f"Processing frame {k + 1} / {nframes}")
    write("traj.xyz", atoms, append = True)
    k += 1

