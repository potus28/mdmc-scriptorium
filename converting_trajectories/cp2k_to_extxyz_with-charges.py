import sys
import pandas as pd
from ase.units import Ha, bar, Bohr, AUT
from ase.data import atomic_masses, atomic_numbers


def read_voronoi_prp(filename):
    with open(filename) as fh:
        natoms = fh.readline()
        _, _, step, _, time, _ = fh.readline().split()

        cell = fh.readline().split()
        del cell[0]
        del cell[0]

        column_names = fh.readline().split()
        del column_names[0]

    return pd.read_table(filename, skiprows=4, names=column_names, sep='\s+')



file_prefix = sys.argv[1] # cp2k-md
runiter = sys.argv[2] # 1
simdir = sys.argv[3] # .

cell_file = f"{file_prefix}-{runiter}.cell"
ener_file = f"{file_prefix}-{runiter}.ener"
traj_file = f"{file_prefix}-pos-{runiter}.xyz"
frc_file = f"{file_prefix}-frc-{runiter}.xyz"
stress_file = f"{file_prefix}-{runiter}.stress"
vel_file = f"{file_prefix}-vel-{runiter}.xyz"
out_file = "traj.xyz"


cell_file = f"{simdir}/{cell_file}"
ener_file = f"{simdir}/{ener_file}"
traj_file = f"{simdir}/{traj_file}"
frc_file = f"{simdir}/{frc_file}"
stress_file = f"{simdir}/{stress_file}"
vel_file = f"{simdir}/{vel_file}"
out_file = f"{simdir}/{out_file}"

with open(ener_file) as fh_ener:
    for line in fh_ener:
        pass
    last_line = line.split()
    nframes = int(last_line[0])

fh_cell = open(cell_file, "r")
fh_ener = open(ener_file, "r")
fh_traj = open(traj_file, "r")
fh_frc = open(frc_file, "r")
fh_stress = open(stress_file, "r")
fh_vel = open(vel_file, "r")
fh_out = open(out_file, "w")

# The first lines of the stress, ener, cell file is a comment
fh_cell.readline()
fh_ener.readline()
fh_stress.readline()

# Main loop
# nframes = 1  # For debugging
for step in range(nframes):
    natoms = int(fh_traj.readline())
    comment = fh_traj.readline()
    natoms = int(fh_vel.readline())
    comment = fh_vel.readline()
    natoms = int(fh_frc.readline())
    comment = fh_frc.readline()

    sys_cell = fh_cell.readline().split()
    sys_ener = fh_ener.readline().split()
    sys_stress = fh_stress.readline().split()

    Ax = float(sys_cell[2])
    Ay = float(sys_cell[3])
    Az = float(sys_cell[4])
    Bx = float(sys_cell[5])
    By = float(sys_cell[6])
    Bz = float(sys_cell[7])
    Cx = float(sys_cell[8])
    Cy = float(sys_cell[9])
    Cz = float(sys_cell[10])

    # CP2K Uses the opposite sign convention for stress
    Sxx = -float(sys_stress[2]) * bar
    Sxy = -float(sys_stress[3]) * bar
    Sxz = -float(sys_stress[4]) * bar
    Syx = -float(sys_stress[5]) * bar
    Syy = -float(sys_stress[6]) * bar
    Syz = -float(sys_stress[7]) * bar
    Szx = -float(sys_stress[8]) * bar
    Szy = -float(sys_stress[9]) * bar
    Szz = -float(sys_stress[10]) * bar

    energy = float(sys_ener[4]) * Ha
    free_energy = energy

    df_voronoi = read_voronoi_prp(f"{simdir}/{file_prefix}-{runiter}_{step}.voronoi")

    s = f"Lattice=\u0022{Ax} {Ay} {Az} {Bx} {By} {Bz} {Cx} {Cy} {Cz}\u0022 "
    s += "Properties=species:S:1:pos:R:3:momenta:R:3:forces:R:3:voronoi_charge:R:1 "
    s += f"energy={energy} "
    s += f"stress=\u0022{Sxx} {Sxy} {Sxz} {Syx} {Syy} {Syz} {Szx} {Szy} {Szz}\u0022 "
    s += f"free_energy={free_energy} "
    s += "pbc=\u0022T T T\u0022"
    fh_out.write(str(natoms) + "\n")
    fh_out.write(s + "\n")

    for atom in range(natoms):
        atom_sym_and_pos = fh_traj.readline().split()
        atom_frc = fh_frc.readline().split()
        atom_vel = fh_vel.readline().split()
        atom_q = df_voronoi["Charge"].iloc[atom]

        sym = atom_sym_and_pos[0]
        rx = atom_sym_and_pos[1]
        ry = atom_sym_and_pos[2]
        rz = atom_sym_and_pos[3]

        m = atomic_masses[atomic_numbers[sym]]

        px = m * float(atom_vel[1]) * Bohr / AUT
        py = m * float(atom_vel[2]) * Bohr / AUT
        pz = m * float(atom_vel[3]) * Bohr / AUT

        fx = float(atom_frc[1]) * Ha / Bohr
        fy = float(atom_frc[2]) * Ha / Bohr
        fz = float(atom_frc[3]) * Ha / Bohr

        s = f"{sym} {rx} {ry} {rz} {px} {py} {pz} {fx} {fy} {fz} {atom_q}"
        fh_out.write(s + "\n")

fh_cell.close()
fh_ener.close()
fh_traj.close()
fh_frc.close()
fh_stress.close()
fh_vel.close()
fh_out.close()
