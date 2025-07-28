
python calculate_dprof.py ../langevin.traj ../indices/H_atoms_indices.npy H
python plot_dprof.py H.dprof.npy H.z.npy 30000 1.008 H 
