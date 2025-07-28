import sys
import numpy as np
from ase.io.trajectory import Trajectory
from ase.visualize import view

traj = Trajectory(sys.argv[1])
idx = np.load(sys.argv[2])
view(traj[0][idx])

