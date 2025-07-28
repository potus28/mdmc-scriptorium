import sys
import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory

traj = Trajectory(sys.argv[1])
selection = np.load(sys.argv[2])
prefix = sys.argv[3]
nbins = 150

bins = np.arange(0, nbins, dtype = int)
nframes = len(traj)

dprof = np.zeros((nframes, nbins))
zs = np.zeros((nframes, nbins))

frame = 0
for atoms in traj:

    if frame % 1000 == 0:
        print(f"Processing frame {frame +1} / {nframes}")

    atoms.wrap()
    a, b, c = atoms.get_cell()
    area = np.linalg.norm(np.cross(a, b))
    cnorm = np.linalg.norm(c)
    z = np.linspace(0, cnorm, nbins+1)

    rz = atoms.get_positions()[:,2]
    rz_selection = rz[selection]

    for binidx in bins:
        z1 = z[binidx]
        z2 = z[binidx +1]
        dz = z2 - z1
        zcenter = 0.5*(z2 + z1)

        dV = area * dz

        n = np.sum((z1 < rz_selection) & (rz_selection < z2))

        drho = n / dV

        dprof[frame][binidx] = drho
        zs[frame][binidx] = zcenter

    frame += 1

np.save(f"{prefix}.dprof.npy",dprof)
np.save(f"{prefix}.z.npy", zs)

