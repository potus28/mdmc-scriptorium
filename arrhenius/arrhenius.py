import numpy as np
import ase
from ase import units
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats, constants
from scipy.optimize import curve_fit, least_squares
from scipy.stats import t
import sys

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

if size == "singlerow halfcolumn":
    figsize=(4.72441, 3.50394) # 120 mm wide

elif size == "doublerow doublecolumn":
    figsize=(7.20472, 7.20472) # 183 mm wide full depth of page is 247 mm


elif size == "singlerow doublecolumn":
    figsize=(7.20472, 3.50394) # 183 mm wide full depth of page is 247 mm


elif size == "singlerow halfcolumn":
    figsize=(4.72441, 3.50394) # 120 mm wide

fig, axs = plt.subplots(dpi=450, nrows=1, ncols=2, figsize=figsize, sharex=False, sharey=False, layout="constrained")


start = int(sys.argv[1])
time_per_frame_ps = float(sys.argv[2]) # 10 (fs / frame) / 1000 (fs/ps) = 0.01
prefix = sys.argv[3]


tinv = lambda p, df: abs(t.ppf(p/2, df))

def confidence_interval(x, res):
    ts = tinv(0.05, len(x)-2)
    slope_confidence = ts * res.stderr
    intercept_confidence = ts * res.intercept_stderr
    return slope_confidence, intercept_confidence

def linear_regression(x, y, equation = None, verbose = True):
    res = stats.linregress(x, y)
    ts = tinv(0.05, len(x)-2)
    if verbose:
        if equation is not None:
            print("y = mx + b")
            print(equation)

        slope_confidence, intercept_confidence = confidence_interval(x, res)
        print(f"R-squared: {res.rvalue**2:.6f}")
        print(f'Slope (95%): {res.slope:.6f} +/- {slope_confidence:.6f}')
        print(f"Intercept (95%): {res.intercept:.6f} +/- {intercept_confidence:.6f}")
        print()
    return res, slope_confidence, intercept_confidence


def calculate_k(infile, start, dt, name = None ,ax = None, custom_args = None):

    if name is not None:
        print(name)

    rxns = np.load(infile)
    time = np.arange(0, len(rxns)) * dt
    analysis = rxns[start:]
    prod_times = time[start:]
    res, kerr, intercept_err = linear_regression(prod_times, analysis, equation = "Nrxn = k * t + b")

    k = res.slope
    b = res.intercept

    if ax is not None:
        if custom_args is not None:
            ax.plot(time, rxns, **custom_args)
            #ax.plot(prod_times, k*prod_times + b , **custom_args)
        else:
            ax.plot(time, rxns)
            #ax.plot(prod_times, k*prod_times + b)

    return k, kerr



def plot_arrhenius(k, kerr, temp, ax, colors, linestyle, marker):
    temps = np.array(temp)
    betas = 1.0 / temps / units.kB
    #lnk = np.log(k)
    #res, Ea_err, intercept_err = linear_regression(betas, k, equation = "ln (k) = ln(A) - Ea * beta")
    #lnA = res.intercept
    #A = np.exp(lnA)
    #Ea = -res.slope

    def arrhenius(beta, A, Ea):
        return A * np.exp(-beta * Ea)

    popt, pcov = curve_fit(arrhenius, betas, k)
    print(popt)

    x = np.linspace(betas[0], betas[-1])
    arrhenius = popt[0] * np.exp(-x * popt[1])
    ax.plot(x, np.log(arrhenius), color = "k", linestyle = linestyle)

    for i in range(len(k)):
        ax.scatter(betas[i], np.log(k[i]), color = colors[i], edgecolors = "k", marker = marker)
    #ax.set_yscale("log")



def plot_rxn_over_time(start, time_per_frame_ps, ax, temps = [450, 600, 750, 900], rxns = ["H-H.com.npy", "H-H.dis.npy"], linestyles = ["solid", "dotted"]):

    plot_legend = True
    ks = np.zeros((len(rxns), len(temps)))
    kerrs = np.zeros((len(rxns), len(temps)))
    for j, r in enumerate(rxns):
        for i, T in enumerate(temps):
            directory = f"{T}_K/coordination"
            infile = f"{directory}/{r}"
            linestyle = linestyles[j]
            color = colors[i]


            if  linestyle == "solid" and plot_legend:
                kji, kerrji = calculate_k(infile, start, time_per_frame_ps, name = f"{T} {r}", ax = ax, custom_args = {"linestyle": linestyle, "color": color,"label": f"{T} K"})
                ks[j][i] = kji
                kerrs[j][i] = kerrji


            else:
                kji, kerrji = calculate_k(infile, start, time_per_frame_ps, name = f"{T} {r}", ax = ax, custom_args = {"linestyle": linestyle, "color": color})
                ks[j][i] = kji
                kerrs[j][i] = kerrji

    return ks, kerrs


temps = [450, 525, 600, 750, 825, 900]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

k, kerr = plot_rxn_over_time(start, time_per_frame_ps, axs[0], temps = temps, rxns = ["H-H.True.com.npy", "H-H.True.dis.npy"], linestyles = ["solid", "dotted"])

print(k[0])
print(k[1])

plot_arrhenius(k[0], kerr[0], temps, axs[1], colors, "solid", "o")
plot_arrhenius(k[1], kerr[1], temps, axs[1], colors, "dotted", "^")

linestyles = ["solid", "dotted"]


for ax in axs:
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

axs[0].set_ylabel(r"Count")
axs[0].set_xlabel(r"$t$ (ps)")

axs[1].set_ylabel(r"ln($k$ (ps$^{-1}$))")
axs[1].set_xlabel(r"1/$k_{B}T$ (1/eV)")

#fig.legend(loc="outside center right")
#fig.legend(loc="outside upper center", ncol = len(temps)/2)
axs[0].legend()
fig.savefig(f"{prefix}.nrxns.jpeg")
fig.savefig(f"{prefix}.nrxns.pdf")
fig.savefig(f"{prefix}.nrxns.eps")


