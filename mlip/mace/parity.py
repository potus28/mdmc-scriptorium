import os, sys, time, random
import numpy as np
from ase import Atoms
from ase.io import iread, read, write
from mace.calculators import MACECalculator

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score




mpl.rcParams["axes.linewidth"] =  1.5
mpl.rcParams["axes.grid"] = False
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.spines.left"] = True
mpl.rcParams["axes.spines.bottom"] = True
mpl.rcParams["axes.spines.top"] = True
mpl.rcParams["axes.spines.right"] = True

mpl.rcParams["xtick.major.width"] = 1.5
mpl.rcParams["ytick.major.width"] = 1.5
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["xtick.minor.width"] = 1.0
mpl.rcParams["ytick.minor.width"] = 1.0
mpl.rcParams["xtick.minor.size"] =  2.5
mpl.rcParams["ytick.minor.size"] =  2.5
mpl.rcParams["xtick.direction"] =  'in'
mpl.rcParams["ytick.direction"] =  'in'
mpl.rcParams["xtick.major.size"] =  5
mpl.rcParams["ytick.major.size"] =  5

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.weight'] ='bold'
mpl.rcParams['font.size'] = 12.0
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42



# 1) Load the test data
traj = iread('../test.xyz')

symbols = {
        "O": ["s", "blue"],
        "Si": ["o", "red"],
        "C": ["^", "green"],
        "H": ["v", "orange"],
        "Sn": ["*", "yellow"],
        }

# 2) Load the MACE model

calc = MACECalculator(
        model_paths=["mace-314_swa.model"],
        models = None, 
        device="cuda",
        default_dtype="float32",
    )



# 3) Get the correlations
energy_correlation = []

force_correlation = {
        "Cs": [],
        "Na": [],
        "In": [],
        "Cl": [],
        "Sb": [],
        }


frame = 0
for dft_atoms in traj:
    print(f"Processing test frame {frame + 1}")
    
    nn_atoms = Atoms(
            dft_atoms.get_chemical_symbols(),
            dft_atoms.get_positions(),
            cell = dft_atoms.get_cell(),
            pbc = dft_atoms.get_pbc()
            )
    nn_atoms.calc = calc
    
    energy_correlation.append((dft_atoms.info['REF_energy']  , nn_atoms.get_potential_energy()))
    
    for element, dft_force, nn_force in zip(dft_atoms.get_chemical_symbols(), dft_atoms.get_array('REF_forces'), nn_atoms.get_forces()):
        force_correlation[element].append((dft_force, nn_force))

    frame += 1

energy_correlation = np.array(energy_correlation)
np.save("energy_correlation.npy", energy_correlation)

for key, value in force_correlation.items():
    force_correlation[key] = np.array(value)
    np.save(f"{key}_force_correlation.npy",force_correlation[key])


'''
fig, axs = plt.subplots(
    nrows = 1, ncols = 2, 
    figsize=(6.5, 3.5), dpi=450, 
    sharex = False, sharey = False, 
    layout="constrained"
    )




for ax, correlation, title in zip(axs, [energy_correlation, force_correlation],['Energy (eV)',' Force (eV/ )']):
    
    if title != "Force":
        x = correlation[:,0]
        y = correlation[:,1]
        ax.scatter(x,y, edgecolors = "k")
        R_square = r2_score(x, y) 
        print(f" {title} $R^2$ = {R_square:.2f}")

    else:
        for element, value in symbols.items():
            x = correlation[element][:,0]
            y = correlation[element][:,1]
            ax.scatter(x,y, label = element, marker = value[0], c = value[1], edgecolors = 'k')
            R_square = r2_score(x, y) 
            print(f" {title} {element} $R^2$ = {R_square:.2f}")

        ax.legend()
    #ax.text(0.1, 0.9,f"$R^2$ = {R_square:.2f}", transform=ax.transAxes)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]), 
        np.max([ax.get_xlim(), ax.get_ylim()]),  
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(title)

fig.savefig('parity.png')
fig.savefig('parity.tiff')
fig.savefig('parity.pdf')
'''

