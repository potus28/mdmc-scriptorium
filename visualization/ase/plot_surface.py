import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import constants
from scipy.optimize import curve_fit, least_squares
from cycler import cycler

from ase import Atoms
from ase.io import read, write
from ase.visualize.plot import plot_atoms


mpl.rcParams["axes.linewidth"] =  2.0
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
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


size="singlerow singlecolumn"

if size == "singlerow singlecolumn":
    figsize=(3.50394, 3.50394) # 89 mm wide

elif size == "doublerow doublecolumn":
    figsize=(7.20472, 7.20472) # 183 mm wide full depth of page is 247 mm

elif size == "singlerow halfcolumn":
    figsize=(4.72441, 3.50394) # 120 mm wide

fig, ax = plt.subplots(dpi=450, nrows=1, ncols=1, figsize=figsize, sharex=True, sharey=False, layout="constrained")


atoms_tmp = read("../langevin.traj", index = -1)
del atoms_tmp[[atom.index for atom in atoms_tmp if atom.symbol == "Fr"]]

groupi = np.load(sys.argv[1])
prefix = sys.argv[2]


dz = 2.0

rz = atoms_tmp.get_positions()[:,2]
zavg = np.mean(rz[groupi])
zlow = zavg - dz
zhigh = zavg + dz

indices = [atom.index for atom in atoms_tmp if zlow < atom.position[2] and atom.position[2] < zhigh or atom.symbol == "C" or atom.symbol == "Mo"]


atoms = Atoms(atoms_tmp.get_chemical_symbols(), atoms_tmp.get_positions(), cell = atoms_tmp.get_cell(), pbc = atoms_tmp.get_pbc())



colors_dict = {"Mo": "#F478E3", "C": "#4C4C4C", "H": "#FFFFFF", }
colors = []
for sym in atoms.get_chemical_symbols():
    colors.append(colors_dict[sym])



slab = atoms[indices]
slab.wrap()
plot_atoms(slab, ax, radii=1.0, colors = colors, rotation=('0x,0y,0z'))

ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))


ax.set_ylabel(r"$y$ (Å)")
ax.set_xlabel(r"$x$ (Å)")



fig.savefig(f"{prefix}.surface.jpeg")
fig.savefig(f"{prefix}.surface.pdf")
fig.savefig(f"{prefix}.surface.eps")

