import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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


size="doublecolumn doublerow"

if size == "singlecolumn singlerow":
    figsize=(3.50394, 3.50394) # 89 mm wide

elif size == "doublecolumn singlerow":
    figsize=(7.20472, 3.50394) # 183 mm wide full depth of page is 247 mm

elif size == "doublecolumn doublerow":
    figsize=(7.20472, 7.20472) # 183 mm wide full depth of page is 247 mm

elif size == "singlecolumn doublerow":
    figsize=(3.50394, 7.20472) # 183 mm wide full depth of page is 247 mm

elif size == "halfcolumn":
    figsize=(4.72441, 4.72441) # 120 mm wide


def plot_equil(directory, prefix, fig, ax):

    cycles = np.load(f"{directory}/{prefix}.out.box1.prp.MC_SWEEP.npy")
    energy_1 = np.load(f"{directory}/{prefix}.out.box1.prp.Energy_Total.npy")
    energy_2 = np.load(f"{directory}/{prefix}.out.box2.prp.Energy_Total.npy")
    rho_1 = np.load(f"{directory}/{prefix}.out.box1.prp.Mass_Density.npy")
    rho_2 = np.load(f"{directory}/{prefix}.out.box2.prp.Mass_Density.npy")

    ax[0][0].plot(cycles, rho_1)
    ax[0][1].plot(cycles, rho_2)
    ax[1][0].plot(cycles, energy_1)
    ax[1][1].plot(cycles, energy_2)

    ax[1][0].set_xlabel("MC Cycle, $n$")
    ax[1][1].set_xlabel("MC Cycle, $n$")
    ax[0][0].set_ylabel(r"$\rho$ (kg/m$^3$)")
    ax[1][0].set_ylabel("$U$ (kJ/mol)")

    ax[1][0].xaxis.set_major_locator(plt.MaxNLocator(4))
    ax[1][1].xaxis.set_major_locator(plt.MaxNLocator(4))

    ax[0][0].yaxis.set_major_locator(plt.MaxNLocator(5))
    ax[0][1].yaxis.set_major_locator(plt.MaxNLocator(5))
    ax[1][0].yaxis.set_major_locator(plt.MaxNLocator(5))
    ax[1][1].yaxis.set_major_locator(plt.MaxNLocator(5))

    #fig.legend(loc="outside center right")

    ax[0][1].set_ylim(4, 10)
    ax[1][1].set_ylim(-10, 1600)



    fig.savefig(f"{prefix}.tiff")
    fig.savefig(f"{prefix}.jpeg")
    fig.savefig(f"{prefix}.pdf")


fig, ax = plt.subplots(dpi=450, nrows=2, ncols=2, figsize=figsize, sharex=True, sharey=False, layout="constrained")
plot_equil("..", "gemc_npt", fig, ax)

