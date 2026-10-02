#!/usr/bin/env python3
"""
Build, view, and export a slab cut from an arbitrary CIF file.

Uses ASE's general surface-cutting algorithm (ase.build.surface), which
works for any bulk structure — not just simple cubic metals — so it
should handle whatever CIF you throw at it (alloys, oxides, etc.),
as long as the CIF gives the conventional (not primitive) cell.

Usage:
    python slab_viz.py structure.cif -m 1 1 1 -l 4 -v 15.0

    python slab_viz.py structure.cif --miller 1 0 0 --layers 6 \
        --vacuum 12 --output my_slab.xyz --no-gui
"""

import argparse
from pathlib import Path

from ase.io import read, write
from ase.build import surface
from ase.visualize import view


def build_slab(cif_path, miller, layers, vacuum, periodic=False):
    """Read a CIF and cut a slab along the given Miller indices.

    Parameters
    ----------
    cif_path : str or Path
        Path to the input CIF file.
    miller : tuple of 3 ints
        Miller indices (h, k, l) defining the surface normal.
    layers : int
        Number of equivalent layers in the slab.
    vacuum : float
        Vacuum added on EACH side of the slab, in Angstrom
        (ASE's convention: total vacuum = 2 * vacuum).
    periodic : bool
        Whether to make the slab periodic along the surface normal too.
        Default False (standard for slab calculations).

    Returns
    -------
    ase.Atoms
        The slab, centered in its cell along the vacuum direction.
    """
    bulk = read(cif_path)
    slab = surface(bulk, miller, layers, vacuum=vacuum, periodic=periodic)
    slab.center(axis=2)
    return slab


def main():
    parser = argparse.ArgumentParser(
        description="Cut a slab from a CIF file, view it, and write it to XYZ."
    )
    parser.add_argument("cif", type=str, help="Path to input CIF file")
    parser.add_argument(
        "-m", "--miller", type=int, nargs=3, default=[1, 0, 0],
        metavar=("H", "K", "L"),
        help="Miller indices, e.g. -m 1 1 1 (default: 1 0 0)",
    )
    parser.add_argument(
        "-l", "--layers", type=int, default=4,
        help="Number of slab layers (default: 4)",
    )
    parser.add_argument(
        "-v", "--vacuum", type=float, default=10.0,
        help="Vacuum in Angstrom added on EACH side (default: 10.0; "
             "ASE doubles this for the total gap between periodic images)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output XYZ filename (default: <cif_stem>_<hkl>_slab.xyz)",
    )
    parser.add_argument(
        "--periodic", action="store_true",
        help="Make the slab periodic along the surface normal (default: off)",
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Skip launching the ASE GUI viewer (just write the XYZ file)",
    )
    args = parser.parse_args()

    miller = tuple(args.miller)

    slab = build_slab(
        args.cif, miller, args.layers, args.vacuum, periodic=args.periodic
    )

    print(f"Read bulk from: {args.cif}")
    print(f"Miller indices: {miller}")
    print(f"Slab: {len(slab)} atoms, formula {slab.get_chemical_formula()}")
    print(f"Cell (a, b, c, alpha, beta, gamma): {slab.cell.cellpar()}")

    hkl_str = "".join(str(i) for i in miller)
    out_name = args.output or f"{Path(args.cif).stem}_{hkl_str}_slab.xyz"
    write(out_name, slab)
    print(f"Wrote slab to: {out_name}")

    if not args.no_gui:
        view(slab)


if __name__ == "__main__":
    main()
