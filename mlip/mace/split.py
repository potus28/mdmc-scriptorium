import sys
import subprocess
import numpy as np
from ase.io import read, iread, write

traj = read('db.xyz', '0:-1:10')
split = int(0.9 * len(traj))

np.random.shuffle(traj)
write('tmp.xyz', traj[:split])
write('test.xyz', traj[split:-1])

subprocess.run("cat isolated_Cl.xyz  isolated_Cs.xyz  isolated_In.xyz  isolated_Na.xyz  isolated_Sb.xyz tmp.xyz > train.xyz", shell=True)
subprocess.run("rm ./tmp.xyz", shell = True)

