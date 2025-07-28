#!/bin/bash


startframe=25000
endframe=250001
logfreq=100

trajectory=../langevin.traj
reference_mol=../../../reference_structures/molecules/H2O.xyz
mol_indices=../indices/sol_group.npy

timestep=0.004
sigma=2
temperature=temperature.npy
eMD=energy.npy

python decompose_velocities.py $trajectory $mol_indices $startframe  $endframe $logfreq
python two_phase_dos.py $trajectory $timestep $sigma $reference_mol $temperature $eMD $startframe $endframe > 2pt.out
python plot_2pt.py

