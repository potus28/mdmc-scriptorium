import os
import sys
import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory
from mace.calculators import MACECalculator

from ase import Atoms, units

np.random.seed(314)

traj = Trajectory(sys.argv[1]) # ../langevin.traj
nframes = len(traj)

basedir = sys.argv[2]  #'/scratch/wwilson/NSF/TMC/MoC/hydrogen-disociation/d-MoC/training/mace/grid/N2_v2_r5_l1_f128'
calc = MACECalculator(
        model_paths=[
            f"{basedir}/model-314_swa.model",
            f"{basedir}/model-914_swa.model",
            f"{basedir}/model-42_swa.model"
            ],
        device='cuda',
        default_dtype="float32"
        )


en = np.zeros((len(traj), 3))
unc = np.zeros(len(traj))

frame = 0
print(f"Active Learning Post Processing on {nframes} frames")
print(f"FrameIdx Uncertainty")
for atoms in traj:
    atoms_eval = Atoms(atoms.get_chemical_symbols(), atoms.get_positions(), cell = atoms.get_cell(), pbc = atoms.get_pbc())
    atoms_eval.calc = calc

    energies = atoms_eval.get_potential_energies() * 1000.0 / len(atoms_eval)
    en[frame] = energies
    unc[frame] = np.var(energies)
    print(frame, unc[frame])
    frame += 1

np.save("energies_meVperatom.npy", en)
np.save("uncertainty.npy", unc)

