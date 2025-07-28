import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams["axes.linewidth"] =  1.5
mpl.rcParams["axes.grid"] = False

mpl.rcParams['axes.titleweight'] = 'bold'
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


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']



fig, ax = plt.subplots(nrows=1, ncols=2, dpi = 600, figsize=(6.5, 3.5), layout="constrained")

for i, label, linestyle in zip(["Mo_110", "Pt_111", "d-MoC_001"], ["Mo (110)", "Pt (111)", "$\delta$-MoC (001)"], ["solid", "dashed", "dashdot"]):
    # energy is already referenced to the fermi level
    energy = np.load(f"{i}/energy.npy")
    pdos = np.load(f"{i}/pdos.npy")

    ax[0].plot(energy, pdos, label = label, linestyle=linestyle)


    I = trapezoid(pdos, energy)
    center = trapezoid(pdos * energy, energy) / I
    width = np.sqrt(trapezoid(pdos * (energy - center)**2, energy) / I)

    print(i)
    print(center)
    print(width)


    x_axis = np.arange(-15, 10, 0.01)
    ax[1].plot(x_axis, norm.pdf(x_axis, center, width), linestyle=linestyle)



ax[0].set_xlabel('$E - E_f$ (eV)')
ax[0].set_ylabel('Surface atom d-projected DOS')

ax[1].set_xlabel('$E - E_f$ (eV)')
ax[1].set_ylabel(r"$P(E - E_f | \mu, \sigma)$")

ax[0].set_xlim((-15, 10))
ax[1].set_xlim((-15, 10))



ax[0].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
ax[0].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
ax[0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))


ax[1].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
ax[1].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
ax[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
ax[1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

fig.legend(loc = "outside upper center", ncols = 3)
fig.savefig('ddos.png', transparent = True)


