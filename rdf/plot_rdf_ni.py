import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rcParams["axes.linewidth"] =  1.5
mpl.rcParams["axes.grid"] = False
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.spines.left"] = True
mpl.rcParams["axes.spines.bottom"] = True
mpl.rcParams["axes.spines.top"] = True
mpl.rcParams["axes.spines.right"] = True

mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["xtick.direction"] =  'in'
mpl.rcParams["ytick.direction"] =  'in'

mpl.rcParams["xtick.major.width"] = 1.5
mpl.rcParams["ytick.major.width"] = 1.5
mpl.rcParams["xtick.major.size"] =  5
mpl.rcParams["ytick.major.size"] =  5
mpl.rcParams["xtick.minor.width"] = 1.0
mpl.rcParams["ytick.minor.width"] = 1.0
mpl.rcParams["xtick.minor.size"] =  2.5
mpl.rcParams["ytick.minor.size"] =  2.5

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.weight'] ='bold'
mpl.rcParams['font.size'] = 12.0
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42

x = np.load("xbins.npy")
y1 = np.load("rdf.npy")
y2 = np.load("number_integral.npy")

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(3.5, 3.5), dpi=600, sharex = True, constrained_layout = True)
axins = inset_axes(axs, width="50%", height="50%", loc="upper right")

axs.hlines(xmin=0, xmax=x[-1], y=1.0, color="k", linestyles=':')
axs.plot(x, y1)



axins.plot(x, y2)


axs.set_ylabel("g($r$)")
axs.set_xlabel("$r$ (Å)")

axins.set_ylabel("N($r$)")


fig.savefig("rdf.png")
#plt.show()

