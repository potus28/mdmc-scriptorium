import sys
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


size="singlerow singlecolumn"

if size == "singlerow singlecolumn":
    figsize=(3.50394, 3.50394) # 89 mm wide

elif size == "doublerow doublecolumn":
    figsize=(7.20472, 7.20472) # 183 mm wide full depth of page is 247 mm

elif size == "singlerow halfcolumn":
    figsize=(4.72441, 3.50394) # 120 mm wide


fig, ax = plt.subplots(dpi=600, nrows=1, ncols=1, figsize=figsize, sharex=True, sharey=False, layout="constrained")

r = np.load("r.npy") # x, y, and z
x = r[:,0]
y = r[:,1]

H, xedges, yedges = np.histogram2d(x, y, bins=100, density = True)
H = H.T
X, Y = np.meshgrid(xedges, yedges)

xcenters = (xedges[:-1] + xedges[1:]) / 2.0
ycenters = (yedges[:-1] + yedges[1:]) / 2.0

im = ax.pcolormesh(xcenters, ycenters, H, cmap = "RdBu", shading='gouraud')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"P($x$,$y$)")

ax.set_ylabel(r"$y$ (Å)")
ax.set_xlabel(r"$x$ (Å)")

ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

fig.savefig(f"heatmap.png")


