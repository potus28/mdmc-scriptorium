
python tetrahedral-otf-select.py ../langevin.traj ../../indices/zeolite_Sn_atoms_indices.npy  ../../indices/zeolite_SnO_atoms_indices.npy O 30000 1.46
python tetrahedral.py ../langevin.traj ../../indices/zeolite_Sn_atoms_indices.npy ../../indices/zeolite_SnO_atoms_indices.npy
python tetrahedral-otf.py ../langevin.traj ../../indices/zeolite_Sn_atoms_indices.npy O 30000 1.46


