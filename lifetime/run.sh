python heaveside.py ../langevin.traj ../../indices/solvent_O_atoms_indices.npy ../../indices/zeolite_Sn_atoms_indices.npy 5.0
python mrt.py 0.01 5000 30000
python quickplot_mrt.py > fitmrt.txt

