import sys
import numpy as np
from scipy import constants
from scipy.integrate import cumtrapz
from numba import njit
import matplotlib.pyplot as plt


@njit
def energy_auto_correlation(earray, ncoor):
    acf = np.zeros(ncorr)
    norm = np.zeros(ncorr)
    nframes = len(earray)

    ebar = np.mean(earray)
    evar = np.var(earray)

    earray -= ebar*np.ones(len(earray)) # mean of 0
    n = 0
    for ener in earray: # Loop over time averages
        if (n + 1) % 1000 == 0: print(f"Processed {n+1} / {nframes}")
        maxn = min(nframes, n + ncorr) - n # Prevents from indexing out of bounds when at the end of the array
        e0 = ener # e(t)

        for i in range(maxn):
            en = earray[n + i] # v(t + i * dt)
            acf[i] += en * e0
            norm[i] += 1

        n += 1

    acf /= norm # Dividing here takes the average value of all observations
    acf /= evar # This ensures that the ACF starts at 1

    return acf


prefix = sys.argv[1]
temp_K = 450


box1_energies = np.load(f"../{prefix}.out.box1.prp.Energy_Total.npy")
box2_energies = np.load(f"../{prefix}.out.box2.prp.Energy_Total.npy")

nframes = len(box1_energies)

box1_energies = box1_energies[nframes // 2:]
box2_energies = box2_energies[nframes // 2:]



kB_kJperK = constants.k / 1000.0
beta_one_over_kJpermol = 1.0 / (kB_kJperK * temp_K * constants.Avogadro)

box1_plus_box2 = box1_energies + box2_energies
box1_plus_box2 *= -beta_one_over_kJpermol

# Correlation depth should not be more than 30% of your total trajectory frames
timestep = 10.0
ncorr = 1000

acf = energy_auto_correlation(box1_plus_box2, ncorr)

# Convert correlation depth to time ()
tau = timestep * np.linspace(0, ncorr - 1, ncorr)
np.save(f"{prefix}.tau.npy", tau)
np.save(f"{prefix}.acf.npy", acf)

