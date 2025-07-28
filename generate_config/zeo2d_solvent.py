import numpy as np
from ase import units, Atom, Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.spacegroup import crystal
from ase.build import surface, add_adsorbate, molecule
from ase.visualize import view
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.npt import NPT
from ase.calculators.cp2k import CP2K
import random

s = 12345
random.seed(s)
np.random.seed(s)

def create_BAsite(atoms, nAl=1, rOH=0.95):
    lowensteins_rule = False
    nl = NeighborList(natural_cutoffs(atoms, mult=1), self_interaction=False, bothways=True)
    nl.update(atoms)
    while not(lowensteins_rule):
        si_atoms = [atom for atom in atoms if atom.symbol == "Si"]
        random.shuffle(si_atoms)
        rand_si_indices = []
        # Set the symbols as aluminum atoms
        for _ in range(nAl):
            idx = si_atoms.pop().index
            rand_si_indices.append(idx)
        # Check if Lowenstein's Rule is satisfied
        if nAl == 1:
            lowensteins_rule = True
        else:
            lowensteins_rule = True
            for i in range(0, nAl-1):
                for j in range(i+1, nAl):
                    i_indices, i_offsets = nl.get_neighbors(rand_si_indices[i])
                    j_indices, j_offsets = nl.get_neighbors(rand_si_indices[j])
                    # Make sure that no Al-O-Al bond exists
                    for idx in i_indices:
                        if idx in j_indices:
                            lowensteins_rule = False

    # Now that we are done checking Lowenstein's rule, let's add the hydrogen atoms
    symbols = atoms.get_chemical_symbols()
    for idx in rand_si_indices:
        symbols[idx] = "Al"
    atoms.set_chemical_symbols(symbols)
    nl = NeighborList(natural_cutoffs(atoms, mult=1), self_interaction=False, bothways=True)
    nl.update(atoms)
    al_atoms = [atom for atom in atoms if atom.symbol == "Al"]
    h_atoms = []
    for al in al_atoms:
        al_indices, al_offsets = nl.get_neighbors(al.index)
        al_pos = al.position
        # Pick one of the oxygen atoms at random
        rint = random.randrange(len(al_indices))
        o_index = al_indices[rint]
        o_pos = atoms.positions[o_index]
        # Get the other neighbor of oxygen
        o_indices, o_offsets = nl.get_neighbors(o_index)

        # This O atom has 2 neighbors, one is the Al atom, one is not
        for idx in o_indices:
            if idx != al.index:
                si_index = idx
                si_pos = atoms.positions[si_index]

        # Apply MIC to distances
        h = np.array(atoms.get_cell()).transpose()
        hinv = np.linalg.inv(h)
        s_o = np.matmul(hinv, o_pos.transpose())
        s_al = np.matmul(hinv, al_pos.transpose())
        s_si = np.matmul(hinv, si_pos.transpose())

        s_si_to_o = s_o - s_si
        s_al_to_o = s_o - s_al

        s_si_to_o -= np.round(s_si_to_o, decimals=0)
        s_al_to_o -= np.round(s_al_to_o, decimals=0)

        r_si_to_o = np.matmul(h, s_si_to_o)
        r_al_to_o = np.matmul(h, s_al_to_o)

        #r_si_to_o = o_pos - si_pos
        #r_al_to_o = o_pos - al_pos
        v = r_si_to_o + r_al_to_o
        u = v / np.linalg.norm(v)
        r_h = o_pos + u * rOH
        h_atoms.append(Atom("H", r_h))

    for h in h_atoms:
        atoms += h


def add_molecules(atoms, adsorbate, n=1, tolerance=2.0):
    """Try to insert n number of probe molecules"""

    cell = np.array(atoms.get_cell())
    anorm = np.sum(cell[:,0])
    bnorm = np.sum(cell[:,1])
    cnorm = np.sum(cell[:,2])

    adsorbate.set_cell(None)
    adsorbate.center()
    n_adsorbate_atoms = len(adsorbate)
    for i in range(n):
        print(f"Adding molecule {i}/{n}")
        overlap = True
        while overlap:
            n_atoms = len(atoms)
            phi, theta, psi = 360.0*np.random.rand(3)
            adsorbate.euler_rotate(phi, theta, psi, center="COM")
            r = np.matmul(cell, np.random.rand(3))
            adsorbate.center()
            adsorbate.translate(r)

            # No other atoms are in the box
            if len(adsorbate) - len(atoms)  == len(adsorbate):
                overlap = False

            # Loop over all positions in the box and check for overlap
            else:
                noverlaps = 0
                for r in atoms.positions:
                    for rtest in adsorbate.positions:
                        dx = r[0] -rtest[0]
                        dy = r[1] -rtest[1]
                        dz = r[2] -rtest[2]

                        dx -= anorm*np.rint(dx / anorm)
                        dy -= bnorm*np.rint(dy / bnorm)
                        dz -= cnorm*np.rint(dz / cnorm)

                        dr = np.sqrt(dx**2 + dy**2 + dz**2)
                        if dr < tolerance:
                            noverlaps +=1
                if noverlaps ==  0:
                    overlap = False

        atoms += adsorbate

def build_zeo_surface(atoms, indices=(0,0,1), layers=1, vacuum=10, rSiO = 1.5, rOH = 0.95 ):
    atoms = surface(atoms, indices, layers, vacuum)
    nl = NeighborList(natural_cutoffs(atoms, mult=1), self_interaction=False, bothways=True)
    nl.update(atoms)
    new_atoms = Atoms()
    for atom in atoms:
        indices, offsets = nl.get_neighbors(atom.index)
        if atom.symbol == "O" and len(indices) == 1:
            r_o = atom.position
            v = np.array([0., 0., 0.])
            for i, offset in zip(indices, offsets):
                v += r_o - (atoms.positions[i] + offset @ atoms.get_cell())
            u = v / np.linalg.norm(v)
            r_H = r_o + u * rOH
            new_atoms += Atoms("H", [r_H])

        if atom.symbol == "Si" and len(indices) == 3:
            r_si = atom.position
            v = np.array([0., 0., 0.])
            for i, offset in zip(indices, offsets):
                v += r_si - (atoms.positions[i] + offset @ atoms.get_cell())
            u = v / np.linalg.norm(v)
            r_o = r_si + u * rSiO
            r_h = r_o + u * rOH
            new_atoms += Atoms("OH", [r_o, r_h])

    return atoms + new_atoms


# Create initial structure.
adsorbate = molecule("CH3OH")
probe = read("BO4.xyz")
atoms = read("MFI.cif")
create_BAsite(atoms, nAl=6, rOH=0.95)
atoms = build_zeo_surface(atoms, (0,1,0), layers=1, vacuum=15)
add_adsorbate(atoms, probe, offset=(0.5, 0.5), height=1.5)

# We want 12 molecules from the classical GCMC simulation results at 10 MPa, 473.15 K
# From preliminary AIMD data, in the liquid phase we want a density of approx 64 molec / 4500 Ang^3
# Let the NPT adjust to the real density over the course of the simulation
box = atoms.get_cell()
a = box[0]
b = box[1]
c = np.array([0., 0., 30.])
Vvac = np.dot(a, np.cross(b,c))
Nvac = int(Vvac*64./4500.)
Nbulk = 12

add_molecules(atoms, adsorbate, n=Nbulk+Nvac, tolerance=1.5)


# Randomly jitter the atoms to give nonzero forces in the first frame.
jitter_factor = 0.1
for atom_pos in atoms.positions:
    for coord in range(3):
        atom_pos[coord] += (2 * np.random.random() - 1) * jitter_factor

write("start.xyz", atoms)
write("start.cif", atoms)

view(atoms)
