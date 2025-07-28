import os
from ase.io.trajectory import Trajectory
from ase.io import write

if os.path.exists("total.traj"):
    try:
        os.remove("total.traj")
    except Exception as e:
        pass

traj_dirs = ["run-0", "run-1"]
logfreq = 100
traj_writer = Trajectory("total.traj", "w")

for run, dirname in enumerate(traj_dirs):
    traj = Trajectory(f"../{dirname}/langevin.traj")
    frame = 0
    nframes = len(traj)
    for atoms in traj:

        if frame % logfreq == 0 : print(f"Processing {frame} / {nframes} in {dirname}")
        if run == 0:
            traj_writer.write(atoms)
        else:
            if frame >= 1:
                traj_writer.write(atoms)
        frame += 1


