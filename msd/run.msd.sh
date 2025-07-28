#!/bin/bash

trajectory="../langevin.traj"
observed="../../indices/solvent_O_atoms_indices.npy"
timestep=0.004
startframe=25000 # 100 ps (900 ps of analysis)



python msd.py $trajectory $observed $timestep $startframe
#ython quickplot_msd.py

