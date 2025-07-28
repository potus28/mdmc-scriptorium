import sys
import numpy as np
from ase.io.trajectory import Trajectory
from scipy import constants
from scipy.integrate import cumtrapz
from numba import njit
import matplotlib.pyplot as plt


# tidynamics: A tiny package to compute the dynamics of stochastic and molecular simulations

def select_power_of_two(n):
    """
    Select the closest i such that n<=2**i
    """
    current_exp = int(np.ceil(np.log2(n+1)))
    if n == 2**current_exp:
        n_fft = n
    if n < 2**current_exp:
        n_fft = 2**current_exp
    elif n > 2**current_exp:
        n_fft = 2**(current_exp+1)

    return n_fft


def autocorrelation_1d(data):
    """
    Compute the autocorrelation of a scalar time series.
    """

    N = len(data)
    n_fft = select_power_of_two(N)

    # Pad the signal with zeros to avoid the periodic images.

    R_data = np.zeros(2*n_fft)
    R_data[:N] = data

    F_data = np.fft.fft(R_data)

    result = np.fft.ifft(F_data*F_data.conj())[:N].real/(N-np.arange(N))

    return result[:N]


def msd(pos):
    """Mean-squared displacement (MSD) of the input trajectory using the Fast
    Correlation Algorithm.

    Computes the MSD for all possible time deltas in the trajectory. The numerical results for large
    time deltas contain fewer samples than for small time times and are less accurate. This is
    intrinsic to the computation and not a limitation of the algorithm.

    Args:
        pos (array-like): The input trajectory, of shape (N,) or (N,D).

    Returns:
        : ndarray of shape (N,) with the MSD for successive linearly spaced time
        delays.

    """

    pos = np.asarray(pos)
    if pos.shape[0] == 0:
        return np.array([], dtype=pos.dtype)
    if pos.ndim==1:
        pos = pos.reshape((-1,1))
    N = len(pos)
    rsq = np.sum(pos**2, axis=1)
    MSD = np.zeros(N, dtype=float)

    SAB = autocorrelation_1d(pos[:,0])
    for i in range(1, pos.shape[1]):
        SAB += autocorrelation_1d(pos[:,i])

    SUMSQ = 2*np.sum(rsq)

    m = 0
    MSD[m] = SUMSQ - 2*SAB[m]*N

    MSD[1:] = (SUMSQ - np.cumsum(rsq)[:-1] - np.cumsum(rsq[1:][::-1])) / (N-1-np.arange(N-1))
    MSD[1:] -= 2*SAB[1:]

    return MSD



# Correlation depth should not be more than 30% of your total trajectory frames
observed_atoms = np.load(sys.argv[2])
timestep = float(sys.argv[3]) #0.004
start_frame = int(sys.argv[4]) #30000

traj = Trajectory(sys.argv[1])[start_frame:]
natoms = len(observed_atoms)
nframes = len(traj)
k = 0

positions = np.zeros([nframes, natoms, 3])

for atoms in traj:
    if k % 1000 == 0: print(f"Processing frame {k + 1} / {nframes}")
    r = atoms.get_positions()
    positions[k] = r[observed_atoms]
    k += 1


print(f"Calculating MSD...")
msds_by_particle = np.zeros([nframes, natoms])
for n in range(natoms):
    print(f"Particle {n+1} / {natoms}")
    msds_by_particle[:, n] = msd(positions[:, n, :])


timeseries = msds_by_particle.mean(axis=1)


# Convert correlation depth to time
tau = timestep * np.arange(len(timeseries))

ncoor = len(tau)//3

# Only save the first ~30% of the calculation, we cannot trust a correlation depth past this...
np.save("tau.npy", tau[1:ncoor])
np.save("msd.npy", timeseries[1:ncoor])

