import sys
from ase import Atoms
from ase.io import read
from ase.visualize.plot import plot_atoms
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from scipy import constants
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit, least_squares
from cycler import cycler

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


size="singlerow doublecolumn"

if size == "singlerow singlecolumn":
    figsize=(3.50394, 3.50394) # 89 mm wide

elif size == "singlerow doublecolumn":
    figsize=(7.20472, 3.50394) # 89 mm wide

elif size == "doublerow doublecolumn":
    figsize=(7.20472, 7.20472) # 183 mm wide full depth of page is 247 mm

elif size == "singlerow halfcolumn":
    figsize=(4.72441, 3.50394) # 120 mm wide


#figsize=(20, 1) # 89 mm wide
#fig, ax = plt.subplots(dpi=450, nrows=1, ncols=6, figsize=figsize, sharex=True, sharey=True, layout="constrained")



fig, ax = plt.subplots(dpi=450, nrows=3, ncols=6, figsize= 2 * np.array([6.4, 2.8]), sharex=False, sharey=False, layout="constrained")

def plot_heatmap(rfile, fig, ax, prefix="Hstar", xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0, xlabel = False, ylabel = False, cbar = False, title = None, density=True):
    r = np.load(rfile) # x, y, and z
    x = r[:,0]
    y = r[:,1]

    if density:
        H, xedges, yedges = np.histogram2d(x, y, bins=100, range = [[xmin, xmax], [ymin, ymax]], density = True)
    else:
        H, xedges, yedges = np.histogram2d(x, y, bins=100, range = [[xmin, xmax], [ymin, ymax]], density = False)

    H = H.T
    X, Y = np.meshgrid(xedges, yedges)

    xcenters = (xedges[:-1] + xedges[1:]) / 2.0
    ycenters = (yedges[:-1] + yedges[1:]) / 2.0

    im = ax.pcolormesh(xcenters, ycenters, H, cmap = "RdBu", shading='gouraud', vmin = 0.0, vmax=0.02, antialiased=True)

    if cbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"P($x$,$y$)")

    if xlabel:
        ax.set_xlabel(r"$x$ (Å)")
    if ylabel:
        ax.set_ylabel(r"$y$ (Å)")

    ax.set_aspect('equal', adjustable='box')
    ax.autoscale()
    ax.set_title(title, loc = "left", fontweight="bold", fontsize=24)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)



def plot_surface(trajfile, ax, groupifile, dz = 2.0, title = None):
    atoms_tmp = read(trajfile, index = -1)
    del atoms_tmp[[atom.index for atom in atoms_tmp if atom.symbol == "Fr"]]
    groupi = np.load(groupifile)

    rz = atoms_tmp.get_positions()[:,2]
    zavg = np.mean(rz[groupi])
    zlow = zavg - dz
    zhigh = zavg + dz

    indices = [atom.index for atom in atoms_tmp if zlow < atom.position[2] and atom.position[2] < zhigh or atom.symbol == "C" or atom.symbol == "Mo"]

    atoms = Atoms(atoms_tmp.get_chemical_symbols(), atoms_tmp.get_positions(), cell = atoms_tmp.get_cell(), pbc = atoms_tmp.get_pbc())

    slab = atoms[indices]
    slab.wrap()

    colors_dict = {"Mo": "#F478E3", "C": "#4C4C4C", "H": "#FFFFFF", }
    colors = []
    for sym in slab.get_chemical_symbols():
        colors.append(colors_dict[sym])

    #plot_atoms(slab, ax, radii=1.0, colors = colors, rotation=('0x,0y,0z'))

    r = slab.get_positions()
    rx = r[:,0]
    ry = r[:,1]
    syms = slab.get_chemical_symbols()
    Mo_indices = [atom.index for atom in slab if atom.symbol == "Mo"]

    H_indices = [atom.index for atom in slab if atom.symbol == "C"]
    C_indices = [atom.index for atom in slab if atom.symbol == "H"]

    ax.scatter(rx[Mo_indices], ry[Mo_indices], marker="o", c = colors_dict["Mo"])
    ax.scatter(rx[C_indices], ry[C_indices], marker="o", c = colors_dict["Mo"])
    ax.scatter(rx[H_indices], ry[H_indices], marker="o", c = colors_dict["Mo"])

    ax.set_title(title, loc = "left", fontweight="bold", fontsize=24)
    #ax.set_axis_off()


# 1) Get the x-y limits to build the heatmap
atoms = read('start.xyz')
cell = atoms.get_cell()

xmin = 0.0
xmax = np.sum(cell[:,0])

ymin = 0.0
ymax = np.sum(cell[:,1])

plot_heatmap("450_K/H-heatmap/Hstar_Mo.r.npy", fig, ax[0][0], prefix="Hstar_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "a)")
plot_heatmap("525_K/H-heatmap/Hstar_Mo.r.npy", fig, ax[0][1], prefix="Hstar_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "b)" )
plot_heatmap("600_K/H-heatmap/Hstar_Mo.r.npy", fig, ax[0][2], prefix="Hstar_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "c)" )
plot_heatmap("750_K/H-heatmap/Hstar_Mo.r.npy", fig, ax[0][3], prefix="Hstar_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "d)" )
plot_heatmap("825_K/H-heatmap/Hstar_Mo.r.npy", fig, ax[0][4], prefix="Hstar_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "e)" )
plot_heatmap("900_K/H-heatmap/Hstar_Mo.r.npy", fig, ax[0][5], prefix="Hstar_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "f)" )


plot_heatmap("450_K/H-heatmap/H2star_Mo.r.npy", fig, ax[1][0], prefix="H2star_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "g)", density = False )
plot_heatmap("525_K/H-heatmap/H2star_Mo.r.npy", fig, ax[1][1], prefix="H2star_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "h)", density = False )
plot_heatmap("600_K/H-heatmap/H2star_Mo.r.npy", fig, ax[1][2], prefix="H2star_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "i)", density = True )
plot_heatmap("750_K/H-heatmap/H2star_Mo.r.npy", fig, ax[1][3], prefix="H2star_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "j)", density = True )
plot_heatmap("825_K/H-heatmap/H2star_Mo.r.npy", fig, ax[1][4], prefix="H2star_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "k)" )
plot_heatmap("900_K/H-heatmap/H2star_Mo.r.npy", fig, ax[1][5], prefix="H2star_Mo", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title = "l)" )

plot_surface("450_K/langevin.traj", ax[2][0], "450_K/indices/surface_Mo_atoms_top_indices.npy", title = "m)" )
plot_surface("525_K/langevin.traj", ax[2][1], "525_K/indices/surface_Mo_atoms_top_indices.npy", title = "n)")
plot_surface("600_K/langevin.traj", ax[2][2], "600_K/indices/surface_Mo_atoms_top_indices.npy", title = "o)")
plot_surface("750_K/langevin.traj", ax[2][3], "750_K/indices/surface_Mo_atoms_top_indices.npy", title = "p)")
plot_surface("825_K/langevin.traj", ax[2][4], "825_K/indices/surface_Mo_atoms_top_indices.npy", title = "q)")
plot_surface("900_K/langevin.traj", ax[2][5], "900_K/indices/surface_Mo_atoms_top_indices.npy", title = "r)")




#python plot_xyheatmap.py H_monolayer_C.r.npy H_monolayer_C
#python plot_xyheatmap.py H_monolayer_Mo.r.npy H_monolayer_Mo

#python plot_xyheatmap.py Hstar_Mo.r.npy Hstar_Mo
#python plot_xyheatmap.py H2star_Mo.r.npy H2star_Mo

#python plot_xyheatmap.py Hstar_C.r.npy Hstar_C
#python plot_xyheatmap.py H2star_C.r.npy H2star_C


fig.savefig(f"heatmap.jpeg")
fig.savefig(f"heatmap.pdf")
fig.savefig(f"heatmap.eps")
