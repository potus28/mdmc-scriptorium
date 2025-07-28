import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import constants
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

fig, ax = plt.subplots(dpi=450, nrows=1, ncols=1, figsize=figsize, sharex=True, sharey=False, layout="constrained")


#dprof = np.load(sys.argv[1]) # natoms / angstrom^3
#z = np.load(sys.argv[2]) # angstrom
#start = int(sys.argv[3]) #30000
#MW = float(sys.argv[4])  # g/mol
#prefix = sys.argv[5]


def plot_dprof(directory,ax, args):
    dprof = np.load(f"{directory}/H.dprof.npy")
    z = np.load(f"{directory}/H.z.npy")
    
    start = 30000
    MW = 1.008


    angstrom_to_m = 1E-10
    g_to_kg = 1E-3
    natoms_to_kg = MW  * g_to_kg / constants.Avogadro
    cf = natoms_to_kg / (angstrom_to_m)**3

    dprof_avg = np.mean(dprof[start:], axis = 0) * cf
    z_avg = np.mean(z[start:], axis = 0)
    ax.plot(dprof_avg, z_avg, **args)


plot_dprof("450_K/dprof", ax, {"linestyle": 'solid', 'label': "450 K", 'marker': "o", 'markevery': 30})
plot_dprof("525_K/dprof", ax, {"linestyle": 'solid', 'label': "525 K", 'marker': "s", 'markevery': 30})
plot_dprof("600_K/dprof", ax, {"linestyle": 'solid', 'label': "600 K", 'marker': 'v', 'markevery': 30})
plot_dprof("750_K/dprof", ax, {"linestyle": 'solid', 'label': "750 K", 'marker': '^', 'markevery': 30})
plot_dprof("825_K/dprof", ax, {"linestyle": 'solid', 'label': "825 K", 'marker': 'D', 'markevery': 30})
plot_dprof("900_K/dprof", ax, {"linestyle": 'solid', 'label': "900 K", 'marker': 'X', 'markevery': 30})

ax.set_ylabel(r"$z$-coordinate (Å)")
ax.set_xlabel(r"$\rho$ (kg / m$^3$)")

ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

#ax.set_xlim((-20, 650))

ax.legend()


prefix = "H"
fig.savefig(f"{prefix}.dprof.jpeg")
fig.savefig(f"{prefix}.dprof.pdf")
fig.savefig(f"{prefix}.dprof.eps")



