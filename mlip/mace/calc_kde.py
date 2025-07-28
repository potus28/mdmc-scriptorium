import numpy as np
from ase import units
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from scipy.stats import linregress,t,gaussian_kde
from sklearn.metrics import r2_score




def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between_vectors(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    value = np.arccos(np.clip(np.dot(v1_u, v2_u), -1, 1))
    if np.isnan(value):
        return -1
    else:
        return value
def get_kde(x, y):
    xy = np.vstack([x, y])
    KDE = gaussian_kde(xy)
    z = KDE(xy)
    return z


energy_correlation = np.load('energy_correlation_peratom.npy')

dft_energy = energy_correlation[:,0]
nn_energy = energy_correlation[:,1]


dft_force = np.load('dft_forces.npy')
nn_force = np.load('nn_forces.npy')

dft_force_mag = np.linalg.norm(dft_force, axis=1)
nn_force_mag = np.linalg.norm(nn_force, axis=1)

angle_error = np.zeros(dft_force_mag.shape)
for index in range(len(dft_force_mag)):
    angle_error[index] = angle_between_vectors(dft_force[index], nn_force[index]) / np.pi * 180


print("Energy Residuals...")
np.save('gkde-energy_residuals.npy', get_kde(dft_energy, dft_energy - nn_energy))
print("Force Residuals...")
np.save('gkde-force_residuals.npy', get_kde(dft_force_mag, dft_force_mag - nn_force_mag))
print("Force and Angle...")
np.save('gkde-force_angle.npy', get_kde(dft_force_mag, angle_error))



