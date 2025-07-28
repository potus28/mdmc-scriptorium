from ase.io import iread, write
from ase.io.trajectory import Trajectory


aimddir = "/home/wwilson/Projects/DOE/mlip-zeolites/aimd/sn-bea"

freq = 8

outtraj = Trajectory('example.traj', 'a')

traj_files = [
        f"{aimddir}/meoh-9/pureSi/traj.xyz",
        f"{aimddir}/meoh-9/t9/traj.xyz",
        f"{aimddir}/meoh-9/st9/traj.xyz",
        f"{aimddir}/wat-39/pureSi/traj.xyz",
        f"{aimddir}/wat-39/t9/traj.xyz",
        f"{aimddir}/wat-39/st9/traj.xyz",
        f"{aimddir}/zeolite/pureSi/traj.xyz",
        f"{aimddir}/zeolite/t9/traj.xyz",
        f"{aimddir}/zeolite/st9/traj.xyz",
        f"{aimddir}/solvent/wat/traj.xyz",
        f"{aimddir}/solvent/meoh/traj.xyz",
]


db = []
traj = Trajectory("total_db.traj")

for infile in traj_files:
    traj = iread(infile, index =":")
    frame = 0
    for atoms in traj:
        if frame % 8 == 0:
            print(f"{infile} frame {frame}")

            traj.
        frame += 1




