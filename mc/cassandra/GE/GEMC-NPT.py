import mbuild
import foyer
import mosdef_cassandra as mc
from mosdef_cassandra.analysis import ThermoProps
from mosdef_cassandra.utils.get_files import get_example_ff_path, get_example_mol2_path
import unyt as u

from warnings import filterwarnings
filterwarnings('ignore', category=UserWarning)
filterwarnings('ignore', category=DeprecationWarning)
from parmed.exceptions import OpenMMWarning
filterwarnings('ignore', category=OpenMMWarning)

water = mbuild.load(get_example_mol2_path("spce"))
spce = foyer.Forcefield(get_example_ff_path("spce"))
water_ff = spce.apply(water)

ethanol = mbuild.load("CCO", smiles = True)
oplsaa = foyer.forcefields.load_OPLSAA()
ethanol_ff = oplsaa.apply(ethanol)

zeolite_lattice = mbuild.lattice.load_cif('Beta_A_Sn9.cif')
compound_dict = {
    "O" : mbuild.Compound(name="O"),
    "Si" : mbuild.Compound(name="Si"),
    "Sn" : mbuild.Compound(name="Sn")
}

zeolite = zeolite_lattice.populate(x=2, y=2, z=1, compound_dict=compound_dict)

ff_zeo = foyer.Forcefield('trappe_zeo.xml')
zeolite_ff = ff_zeo.apply(zeolite)

liquid_box = mbuild.fill_box(
    compound = ethanol,
    n_compounds = 512,
    #n_compounds = 500,
    box=[3.5, 3.5, 3.5]
    )

#vapor_box = mbuild.fill_box(
#    compound = ethanol,
#    n_compounds = 12,
#    box=[4.5, 4.5, 4.5]
#    )


# Create box and species list
#box_list = [zeolite, liquid_box, vapor_box]
#box_list = [zeolite, vapor_box]
box_list = [zeolite, liquid_box]

species_list = [zeolite_ff, ethanol_ff]

# Since we have an occupied box we need to specify
# the number of each species present in the initial config
mols_in_boxes= [[1, 0], [0, 512]]
#mols_in_boxes= [[500], [12]]

system = mc.System(box_list, species_list, mols_in_boxes = mols_in_boxes)


moveset = mc.MoveSet("gemc_npt", species_list)
moveset.max_volume[0] = 0.0

#moveset = mc.MoveSet("gemc_nvt", species_list)

custom_args = {
     "units": "sweeps",
     "steps_per_sweep": 512,
     "cutoff_style": "cut_tail",
     "vdw_cutoff_box1": 12.0 * u.Angstrom,
     "vdw_cutoff_box2": 12.0 * u.Angstrom,
     "charge_style": "ewald",
     "charge_cutoff_box1": 12.0 * u.Angstrom,
     "charge_cutoff_box2": 12.0 * u.Angstrom,
     "prop_freq": 10,
     "coord_freq": 10,
     "mixing_rule": "geometric",
     "rcut_min": 0.05 * u.Angstrom,
     "seeds": [12345, 6789],
 }


mc.run(
    system=system,
    moveset=moveset,
    run_type="equilibration",
    run_length=50000,
    pressure = 1.0 * u.bar,
    temperature=450.0 * u.K,
    **custom_args,
)


