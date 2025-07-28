import sys
from ase import build, units
from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
from ase.calculators.cp2k import CP2K


atoms_unit_cell = read(sys.argv[1])
nx = 3
ny = 3
nlayers = 6
vacuum = 15.0
miller = (1, 1, 1)
inverted = False


cp2k_inp = '''
&FORCE_EVAL
   STRESS_TENSOR ANALYTICAL
   &DFT
      &QS
        EPS_DEFAULT 1.0E-10
        EXTRAPOLATION USE_PREV_WF
      &END
      &SCF
         EPS_SCF 1.0E-5
         &OUTER_SCF .TRUE.
            EPS_SCF 1.0E-5
            MAX_SCF 50
         &END OUTER_SCF
         &OT .TRUE.
             ALGORITHM  IRAC
             MINIMIZER  DIIS
             N_HISTORY_VEC  7
             PRECONDITIONER  FULL_KINETIC
             PRECOND_SOLVER INVERSE_CHOLESKY
             ROTATION  .TRUE.
             OCCUPATION_PRECONDITIONER  .TRUE.
         &END OT
     &END SCF
      &XC
         &XC_GRID
            XC_DERIV NN10_SMOOTH
            XC_SMOOTH_RHO NN10
         &END XC_GRID
         &VDW_POTENTIAL
            POTENTIAL_TYPE PAIR_POTENTIAL
            &PAIR_POTENTIAL
               TYPE DFTD3(BJ)
               REFERENCE_FUNCTIONAL PBE
               R_CUTOFF 10.0
               CALCULATE_C9_TERM .TRUE.
               VERBOSE_OUTPUT .TRUE.
               PARAMETER_FILE_NAME dftd3.dat
            &END PAIR_POTENTIAL
         &END VDW_POTENTIAL
         &XC_FUNCTIONAL
            &GGA_X_PBE
            &END GGA_X_PBE
            &GGA_C_PBE
            &END GGA_C_PBE
         &END XC_FUNCTIONAL
      &END XC
      &MGRID
         REL_CUTOFF 60
         NGRIDS 5
      &END MGRID
   &END DFT
&END FORCE_EVAL
'''

calc = CP2K(
        basis_set = 'DZVP-MOLOPT-SR-GTH',
        basis_set_file = 'BASIS_MOLOPT',
        charge = 0,
        cutoff = 600 * units.Rydberg,
        inp = cp2k_inp,
        max_scf = 50,
        pseudo_potential = 'GTH-PBE',
        stress_tensor = True,
        uks = True,
        xc = None,
        command="env OMP_NUM_THREADS=1 mpiexec -np 4 cp2k_shell.psmp"
        )

atoms_unit_cell.calc = calc
E_unitcell = atoms_unit_cell.get_potential_energy()
atoms_surface = build.surface(atoms_unit_cell, miller, nlayers, vacuum, periodic=True)

if inverted:
    r = atoms.get_positions()
    r[:,2] *= -1
    atoms.wrap()

mask = [atom.scaled_position[2][2] < 0.5 for atom in atoms_surface]
atoms_surface.set_constraint(FixAtoms(mask=mask))
cell = atoms.get_cell()
area = np.linalg.norm(np.cross(cell[0], cell[1]))

atoms_surface.calc = calc

E_unrelaxed = atoms_surface.get_potential_energy()

dyn = LBFGS(atoms_surface, trajectory="lbfgs.traj", logfile="lbfgs.log")
dyn.run(fmax=0.05)
E_relaxed = atoms_surface.get_potential_energy()

natoms_per_unit_cells = len(atoms_unit_cell)
nucs = len(atoms_surface) // natoms_per_unit_cells

E_cleavage = 0.5*(E_unrelaxed -nucs*E_unitcell)
E_relax = E_relaxed - E_unrelaxed

gamma = (E_cleavage + E_relax)/area

print(f'Surface Energy (eV/angstrom**2) = {gamma}')

