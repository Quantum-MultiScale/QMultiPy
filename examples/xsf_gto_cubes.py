#!/usr/bin/env python3
import argparse
import pathlib

import numpy as np

from qmultipy.io.cube import read_cube, write_cube
from qmultipy.io.xsf import read_xsf
from qmultipy.maps import direct_atomic_to_ao, direct_to_atomic_accurate


def _mol_from_ions(ions, basis):
    from pyscf import gto

    atoms = ions.to_ase()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    atom_lines = "\n".join(
        f"{sym} {pos[0]} {pos[1]} {pos[2]}" for sym, pos in zip(symbols, positions)
    )
    return gto.M(atom=atom_lines, basis=basis, unit="Angstrom")


def _rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


def _fit_gto_full_pruned(field, mol, cell, grid_level=4):
    from pyscf.dft.gen_grid import Grids

    grids = Grids(mol)
    grids.level = grid_level
    grids.build()

    coords = grids.coords
    weights = grids.weights
    frac = cell.scaled_positions(coords)
    mask = np.all((frac >= 0.0) & (frac < 1.0), axis=1)
    coords = coords[mask]
    weights = weights[mask]

    field_atomic = direct_to_atomic_accurate(field, coords)
    mat = direct_atomic_to_ao(mol, field_atomic, coords, weights)
    return mat


def main():
    parser = argparse.ArgumentParser(
        description="Fit XSF density to GTOs, write PySCF + QMultiPy cubes, compare."
    )
    parser.add_argument(
        "--xsf",
        type=pathlib.Path,
        default=pathlib.Path("test/io/DATA/al_random.xsf"),
        help="Input XSF file.",
    )
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path("test/io/DATA"),
        help="Output directory.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="6-311G*",
        help="PySCF basis set.",
    )
    args = parser.parse_args()

    if not args.xsf.is_file():
        raise FileNotFoundError(f"Missing XSF file: {args.xsf}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    ions, field, _ = read_xsf(args.xsf, kind="all", full=True, units="angstrom")
    ions.set_charges(0.0)

    mol = _mol_from_ions(ions, args.basis)
    dm = _fit_gto_full_pruned(field, mol, ions.to_ase().cell, grid_level=4)

    from pyscf.tools import cubegen

    pyscf_cube = args.outdir / f"{args.xsf.stem}_pyscf.cube"
    cubegen.density(mol, str(pyscf_cube), dm, resolution=field.grid.spacings.max())

    qmultipy_cube = args.outdir / f"{args.xsf.stem}_qmultipy.cube"
    write_cube(qmultipy_cube, ions=ions, data=field)

    _, pyscf_field, _ = read_cube(pyscf_cube, kind="all")
    _, qmp_field, _ = read_cube(qmultipy_cube, kind="all")

    points = pyscf_field.grid.r.reshape(3, -1).T
    cell = ions.to_ase().cell
    frac = cell.scaled_positions(points)
    mask = np.all((frac >= 0.0) & (frac < 1.0), axis=1)

    qmp_on_pyscf = np.zeros(points.shape[0])
    qmp_on_pyscf[mask] = direct_to_atomic_accurate(qmp_field, points[mask])
    pyscf_vals = np.asarray(pyscf_field).reshape(-1)
    rms = _rmse(qmp_on_pyscf[mask], pyscf_vals[mask])

    print(f"RMSE QMultiPy cube vs PySCF cube (on PySCF grid): {rms:.6e}")
    print(f"PySCF cube: {pyscf_cube}")
    print(f"QMultiPy cube: {qmultipy_cube}")


if __name__ == "__main__":
    main()
