import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt

def restime_exponential(x, A, B, t1, t2):
    return A*np.exp(-x/t1) + B*np.exp(-x/t2)

def integral_of_restime_exponential(A, B, t1, t2):
    return A*t1 + B*t2

taus = np.load("tau.npy")
restimes = np.load("restime.npy")

pars, cov = curve_fit(f=restime_exponential, xdata = taus, ydata = restimes, p0=[0, 0, 1, 1], bounds = (-np.inf, np.inf), maxfev = 10000)
tau = integral_of_restime_exponential(*pars)

print("A B tau1 tau2")
print(*pars)
print("Mean Residence Time (ps) =", tau)

fig, ax = plt.subplots(dpi=450, figsize=(3.5,3.5),  layout="constrained")
ax.plot(taus, restimes)
ax.plot(taus, restime_exponential(taus, *pars), linestyle = "dotted")

ax.text(
     0.70, 0.7,
     f"MRT = {tau:.3f} ps",
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes
     )



ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))


ax.set_xlabel(r'$\tau$ (ps)')
ax.set_ylabel(r'R($\tau$)')

fig.savefig("mrt.pdf")
fig.savefig("mrt.eps")
fig.savefig("mrt.png")

