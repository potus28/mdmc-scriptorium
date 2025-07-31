import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from cycler import cycler

colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
mpl.rcParams["axes.prop_cycle"] = cycler(color=colors)
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


size="singlecolumn"
height = 3.50394

if size == "singlecolumn":
    figsize=(3.50394, height) # 89 mm wide

elif size == "doublecolumn":
    figsize=(7.20472, height) # 183 mm wide full depth of page is 247 mm

elif size == "halfcolumn":
    figsize=(4.72441, height) # 120 mm wide

fig, ax = plt.subplots(dpi=450, nrows=1, ncols=1, figsize=figsize, sharex=False, sharey=False, layout="constrained")

def exponential(x, r): #function f(x, r) = e^(r*x)
    return np.exp(r*x)

def tau_calc(ac_data, function = exponential): #takes in data and a python function
    x_data = np.arange(len(ac_data))
    pars, cov = curve_fit(f=function, xdata=x_data, ydata=ac_data, p0=[0], bounds=(-np.inf, np.inf))
    return -1/pars #tau = -1/k

def plot_acf(tau_file, acf_file, timestep):
    tau = np.load(tau_file)
    acf = np.load(acf_file)
    tau_fitted = tau_calc(acf) * timestep
    ax.plot(tau, acf, label = "ACF")
    ax.plot(tau, exponential(tau, -1/tau_fitted), label = f"exp(-$n$ / {tau_fitted[0]:.2f})", linestyle = "dotted")


#axs[0].text(-0.15, 0.95, "a)", horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)

plot_acf("gemc.tau.npy", "gemc.acf.npy", timestep = 10.0)

ax.set_xlabel("MC Cycle, $n$")
ax.set_ylabel(r"-$\beta$($U_1$ + $U_2$)")
ax.legend(loc="best")

ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

prefix = "acf"
fig.savefig(f"{prefix}.tiff")
fig.savefig(f"{prefix}.jpeg")
fig.savefig(f"{prefix}.pdf")
