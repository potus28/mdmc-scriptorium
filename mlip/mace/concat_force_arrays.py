import numpy as np

elements = ["Cl", "Cs",  "In", "Na", "Sb"]

for i, e in enumerate(elements):
    if i == 0:
        force_correlation = np.load(f"{e}_force_correlation.npy")
        dft_forces = force_correlation[:, 0]
        nn_forces = force_correlation[:, 1]
    else:
        force_correlation = np.load(f"{e}_force_correlation.npy")
        dft_forces = np.vstack((dft_forces, force_correlation[:, 0]))
        nn_forces = np.vstack((nn_forces, force_correlation[:, 1]))

np.save("dft_forces.npy", dft_forces)
np.save("nn_forces.npy", nn_forces)
