
python compute_rvicinity.py ../langevin.traj ../indices/H_atoms_indices.npy ../indices/surface_C_atoms_indices.npy 2.0 30000 H_monolayer_C
python compute_rvicinity.py ../langevin.traj ../indices/H_atoms_indices.npy ../indices/surface_Mo_atoms_indices.npy 2.0 30000 H_monolayer_Mo

python compute_rvicinity_withcoordination.py ../langevin.traj ../indices/H_atoms_indices.npy ../indices/surface_Mo_atoms_indices.npy 2.0 30000 Hstar_Mo ../coordination/H-H.coordination.npy 0
python compute_rvicinity_withcoordination.py ../langevin.traj ../indices/H_atoms_indices.npy ../indices/surface_Mo_atoms_indices.npy 2.0 30000 H2star_Mo ../coordination/H-H.coordination.npy 1
python compute_rvicinity_withcoordination.py ../langevin.traj ../indices/H_atoms_indices.npy ../indices/surface_C_atoms_indices.npy 2.0 30000 Hstar_C ../coordination/H-H.coordination.npy 0
python compute_rvicinity_withcoordination.py ../langevin.traj ../indices/H_atoms_indices.npy ../indices/surface_C_atoms_indices.npy 2.0 30000 H2star_C ../coordination/H-H.coordination.npy 1

python plot_xyheatmap.py H_monolayer_C.r.npy H_monolayer_C
python plot_xyheatmap.py H_monolayer_Mo.r.npy H_monolayer_Mo

python plot_xyheatmap.py Hstar_Mo.r.npy Hstar_Mo
python plot_xyheatmap.py H2star_Mo.r.npy H2star_Mo

python plot_xyheatmap.py Hstar_C.r.npy Hstar_C
python plot_xyheatmap.py H2star_C.r.npy H2star_C

