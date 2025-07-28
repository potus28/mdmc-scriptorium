#!/bin/bash

trajectory=../langevin.traj

reference=../../indices/zeolite_Sn_atoms_indices.npy
observed=../../indices/solvent_O_atoms_indices.npy
maxd=10.0
nbins=150
start_frame=25000

python rdf_numba.py $trajectory $reference $observed $maxd $nbins $start_frame
