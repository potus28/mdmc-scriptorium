import sys
from ase import Atoms
from ase.io import read
from ase.visualize.plot import plot_atoms
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import matplotlib.cm as cm

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


#figsize=(20, 1) # 89 mm wide
#fig, ax = plt.subplots(dpi=450, nrows=1, ncols=6, figsize=figsize, sharex=True, sharey=True, layout="constrained")

fig, ax = plt.subplots(dpi=600, nrows=2, ncols=2, figsize = (3.5, 2.5), sharex=True, sharey=True, layout="constrained")

def plot_heatmap(rfile, fig, ax, xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0, xlabel = False, ylabel = False, cbar = False, title = None, cmap = "Blues"):
    r = np.load(rfile) # x, y, and z
    x = r[:,0]
    y = r[:,1]

    H, xedges, yedges = np.histogram2d(x, y, bins=100, range = [[xmin, xmax], [ymin, ymax]], density = True)
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)

    H = np.ma.masked_where(H < 0.005, H)

    xcenters = (xedges[:-1] + xedges[1:]) / 2.0
    ycenters = (yedges[:-1] + yedges[1:]) / 2.0

    im = ax.pcolormesh(xcenters, ycenters, H, cmap = cmap, shading='gouraud', vmin = 0.0, vmax=0.02, antialiased=True)

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

    return im

def plot_surface(coordfile, ax, dz = 1.5, title = None, offset = (0.0, 0.0)):
    atoms = read(coordfile)

    colors_dict = {"Mo": "#F478E3", "C": "#4C4C4C", "H": "#FFFFFF", "Pt":"#999999"}
    colors = []
    for sym in atoms.get_chemical_symbols():
        colors.append(colors_dict[sym])

    plot_atoms(atoms, ax, radii=1.0, colors = colors, offset = offset, rotation=('0x,0y,0z'), alpha= 0.45)
    ax.set_title(title, loc = "left", fontweight="bold", fontsize=24)
    ax.set_axis_off()


# 1) Get the x-y limits to build the heatmap
atoms = read('start.xyz')
cell = atoms.get_cell()

xmin = 0.0
xmax = np.sum(cell[:,0])

ymin = 0.0
ymax = np.sum(cell[:,1])


offset = (-1.6, -1.6)
plot_surface("start.xyz", ax[0][0], offset = offset)
plot_surface("start.xyz", ax[0][1], offset = offset)
plot_surface("start.xyz", ax[1][0], offset = offset)
plot_surface("start.xyz", ax[1][1], offset = offset)

imh = plot_heatmap("B3_T500/heatmap/r.npy", fig, ax[0][0],  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmap = "Blues")
imh = plot_heatmap("B3_T750/heatmap/r.npy", fig, ax[0][1],  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmap = "Blues")
imh = plot_heatmap("B3_T1000/heatmap/r.npy", fig, ax[1][0],  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmap = "Blues")
imh = plot_heatmap("B3_T1250/heatmap/r.npy", fig, ax[1][1],  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, cmap = "Blues")

ax[0][0].set_title("a) 500 K", fontweight = "bold")
ax[0][1].set_title("b) 750 K", fontweight = "bold")
ax[1][0].set_title("c) 1000 K", fontweight = "bold")
ax[1][1].set_title("d) 1250 K", fontweight = "bold")

cbarh = fig.colorbar(imh, ax=ax.ravel().tolist())
cbarh.ax.yaxis.set_major_locator(plt.MaxNLocator(4))
cbarh.ax.set_xlabel("NH$_3$")

fig.savefig(f"heatmap.png")

