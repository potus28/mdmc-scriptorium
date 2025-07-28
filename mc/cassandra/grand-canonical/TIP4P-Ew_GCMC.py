#!/usr/bin/env python
# coding: utf-8

# In[67]:


get_ipython().run_line_magic('set_env', 'SHELL=/bin/bash')
get_ipython().run_line_magic('set_env', 'OMP_NUM_THREADS=4')


# In[68]:


import mbuild as mb
import foyer
import mosdef_cassandra as mc
import unyt as u
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import numpy as np

from scipy.stats import linregress

from mosdef_cassandra.analysis import ThermoProps
from mosdef_cassandra.utils.tempdir import temporary_cd
from mosdef_cassandra.utils.get_files import get_example_ff_path, get_example_mol2_path

# Filter some warnings -- to cleanup output for this demo
from warnings import filterwarnings
filterwarnings('ignore', category=UserWarning)
from parmed.exceptions import OpenMMWarning
filterwarnings('ignore', category=OpenMMWarning)
warnings.filterwarnings('ignore')


mpl.rcParams["axes.linewidth"] =  1.5
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
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

get_ipython().run_line_magic('matplotlib', 'inline')



def set_ticks(ax, xlinear=True, ylinear=False, xmax=False, ymax=True, yticks = 5, xticks = 5):
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(6))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(6))
    if xlinear: ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(xticks))
    if xmax: ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(xticks))
    if ylinear: ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(yticks))
    if ymax: ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(yticks))





# In[69]:


#adsorbate = mb.load(get_example_mol2_path("spce"))
#ff = foyer.Forcefield(get_example_ff_path("spce"))


rOH =  0.09526 # nm
rOM = 0.01250 # nm
thetaHOH =  104.52  # degrees

x = thetaHOH * np.pi / 180.0 / 2

adsorbate = mb.Compound()
o = mb.Particle(name='O', pos=[0.0, 0.0, 0.0])
h1 = mb.Particle(name='H', pos=[0.0, rOH * np.cos(x), rOH * np.sin(x)])
h2 = mb.Particle(name='H', pos=[0.0, rOH * np.cos(x), -rOH * np.sin(x)])
m =  mb.Particle(name='_M', pos=[0.0, rOM, 0.0])
adsorbate.add([o, h1, h2, m])
adsorbate.add_bond((o, h1))
adsorbate.add_bond((o, h2))
adsorbate.add_bond((o, m))

ff = foyer.Forcefield("tip4p_ew.xml")


adsorbate_ff = ff.apply(adsorbate)
adsorbate.visualize()


# In[70]:


zeolite_lattice = mb.lattice.load_cif('../Beta_A.cif')
compound_dict = {
    "O" : mb.Compound(name="O"),
    "Si" : mb.Compound(name="Si")
}


nx = 3
ny = 3
nz = 2

zeolite = zeolite_lattice.populate(x=nx, y=ny, z=nz, compound_dict=compound_dict)

trappe_zeo = foyer.Forcefield("../zeo_trappe.xml")
zeolite_ff = trappe_zeo.apply(zeolite)

zeolite.visualize()


# In[71]:


temperature = 373.0 * u.K

custom_args = {
  "rcut_min": 0.5 * u.angstrom,
    "vdw_cutoff": 14.0 * u.angstrom,
    "charge_cutoff": 14.0 * u.angstrom,
    "prop_freq": 1000,
    "coord_freq": 1000,
    "angle_style": ['fixed'], # Only for riged molecules! 
}



mus_adsorbate = np.arange(-58, -48, 2) * u.kJ/u.mol

for mu_adsorbate in mus_adsorbate:
    dirname = f'pure_mu_{mu_adsorbate:.1f}_T_{temperature:.1f}'.replace(" ", "_").replace("/", "-")
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    else:
        pass
    with temporary_cd(dirname):
        species_list = [adsorbate_ff]
        if mu_adsorbate < -36:
            boxl = 10. # nm
        else:
            boxl = 2.5 # nm
        box_list = [mb.Box([boxl,boxl,boxl])]
        system = mc.System(box_list, species_list)
        moveset = mc.MoveSet('gcmc', species_list)
        moveset.prob_regrow = 0.0 # Only for riged molecules! 
        mc.run(
            system=system,
            moveset=moveset,
            run_type="equil",
            run_length=100000,
            temperature=temperature,
            chemical_potentials = [mu_adsorbate],
            **custom_args
        )


# In[72]:


fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 600, figsize = (3.5, 3.5), layout = "constrained")

pressures = []
for mu_adsorbate in mus_adsorbate:
    dirname = f'pure_mu_{mu_adsorbate:.1f}_T_{temperature:.1f}'.replace(" ", "_").replace("/", "-")
    thermo = ThermoProps(dirname + "/gcmc.out.prp")
    pressures.append(np.mean(thermo.prop("Pressure", start=30000)))
    plt.plot(thermo.prop("MC_STEP"), thermo.prop("Pressure"))
    
ax.set_xlabel("MC Step")
ax.set_ylabel("Pressure (bar)")
set_ticks(ax,  xlinear=False, ylinear=False, xmax=True, ymax=True,)

pass


# In[73]:


fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 600, figsize = (3.5, 3.5), layout = "constrained")
slope, intercept, r_value, p_value, stderr = linregress(np.log(pressures).flatten(),y=mus_adsorbate.flatten())



mus_eos = (slope * np.log(pressures) + intercept) * u.kJ/u.mol
ax.plot(mus_eos, pressures, 'b--')

ax.plot(mus_adsorbate, pressures, 'ro')
ax.set_xlabel("Chemical potential (kJ/mol)")
ax.set_ylabel("Fugacity (bar)")
ax.set_yscale('log')


print(slope)
print(intercept)

ax.text(
        0.2, 0.85,
        f"$T$ = 373 K",
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax.transAxes,
    )


ax.text(
        0.2, 0.75,
        f"R$^2$ = {r_value * r_value:.3f}",
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax.transAxes,
    )

pass


# In[74]:


pressures = [
    6000   ,
    22100  ,
    49180  ,
    101325 , # 1 atmosphere
    121800 ,
    316800 ,
    839700 ,
    2243000,
    6000000,
] * u.Pa

mus = (slope * np.log(pressures.in_units(u.bar)) + intercept) * u.kJ/u.mol
for (mu, pressure) in zip(mus, pressures):
    print(f"We will run at mu = {mu:0.2f} to simulate {pressure:0.0f}")



# In[75]:


box_list = [zeolite]
species_list = [zeolite_ff, adsorbate_ff]

mols_in_boxes = [[1,0]]

system = mc.System(
    box_list,
    species_list,
    mols_in_boxes=mols_in_boxes,
)

moveset = mc.MoveSet('gcmc', species_list)
moveset.prob_regrow = 0.0 # Only for riged molecules! 

custom_args = {
  "rcut_min": 0.5 * u.angstrom,
    "vdw_cutoff": 14.0 * u.angstrom,
    "charge_cutoff": 14.0 * u.angstrom,
    "max_molecules": [1, 5000],
    "angle_style": ['fixed', "fixed"],
    "prop_freq": 4000,
    "coord_freq": 4000,
}

for (pressure, mu) in zip(pressures, mus):
    
    dirname = f"zeo_press_{pressure.value:0.0f}_T_{temperature:.1f}"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    else:
        pass
    with temporary_cd(dirname):
        mc.run(
            system=system,
            moveset=moveset,
            run_type="equil",
            run_length=20000000,
            temperature=temperature,
            chemical_potentials = ["none", mu],
            **custom_args
        )


# In[ ]:


fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 600, figsize = (6.5, 3.5), layout = "constrained")
loading = []
for pressure in pressures:
    dirname = f"zeo_press_{pressure.value:0.0f}_T_{temperature:.1f}"
    thermo = ThermoProps(f"{dirname}/gcmc.out.prp")
    n_unitcells = nx*ny*nz
    loading.append(np.mean(thermo.prop("Nmols_2", start=10000000)/n_unitcells))
    ax.plot(thermo.prop("MC_STEP"), thermo.prop("Nmols_2")/n_unitcells, label=f"{pressure:0.0f}")
    
ax.set_title(f"T = {temperature:0.1f}", fontweight="bold")
ax.set_xlabel('MC Step')

ax.set_ylabel('Adsorbate / Unit Cell')
set_ticks(ax,  xlinear=False, ylinear=False, xmax=True, ymax=True,)

fig.legend(loc='outside center right', ncol=1)
pass


# In[ ]:


#published_results = np.genfromtxt('resources/lit_results/tzeo_MFI-methane_308K.txt', skip_header=3)
#plt.plot(published_results[:,0], published_results[:,1], 'bs', markersize=8, label="Siepmann 2013")
fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 600, figsize = (3.5, 3.5), layout = "constrained")
ax.plot(pressures, loading, 'ro-', markersize=8)
#plt.title("Comparison with literature")
ax.set_xlabel("Fugacity (Pa)")
ax.set_ylabel("Loading (molecule/uc)")
ax.set_xscale("log")


# In[ ]:




