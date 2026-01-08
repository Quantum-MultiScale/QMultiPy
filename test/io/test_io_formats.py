import numpy as np

from qmultipy.field import DirectField
from qmultipy.grid import DirectGrid
from qmultipy.ions import Ions
from qmultipy.io.cube import read_cube, write_cube
from qmultipy.io.vasp import read_vasp, write_vasp
from qmultipy.io.xsf import read_xsf, write_xsf
from qmultipy.io.xyz import read_xyz, write_xyz


def _ions_ase():
    symbols = ["H", "O", "H"]
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.75, -0.24]])
    cell = np.diag([8.0, 8.0, 8.0])
    return Ions(symbols=symbols, positions=positions, cell=cell, units="ase")


def _ions_au():
    symbols = ["H", "O", "H"]
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.4, -0.5]])
    cell = np.diag([15.0, 15.0, 15.0])
    return Ions(symbols=symbols, positions=positions, cell=cell, units="au")


def _sorted_symbols_positions(ions):
    atoms = ions.to_ase()
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    order = np.argsort(symbols)
    return symbols[order].tolist(), positions[order], atoms.cell.array


def test_xyz_roundtrip(tmp_path):
    ions = _ions_ase()
    path = tmp_path / "water.xyz"
    write_xyz(path, ions)

    read_ions = read_xyz(path)
    assert read_ions.to_ase().get_chemical_symbols() == ions.to_ase().get_chemical_symbols()
    assert np.allclose(
        read_ions.to_ase().get_positions(), ions.to_ase().get_positions()
    )
    assert np.allclose(read_ions.to_ase().cell.array, ions.to_ase().cell.array)


def test_vasp_poscar_roundtrip(tmp_path):
    ions = _ions_ase()
    path = tmp_path / "POSCAR"
    write_vasp(path, ions, direct=False)

    read_ions = read_vasp(path)
    symbols_ref, pos_ref, cell_ref = _sorted_symbols_positions(ions)
    symbols_read, pos_read, cell_read = _sorted_symbols_positions(read_ions)
    assert symbols_read == symbols_ref
    assert np.allclose(pos_read, pos_ref)
    assert np.allclose(cell_read, cell_ref)


def test_xsf_roundtrip_with_data(tmp_path):
    ions = _ions_au()
    grid = DirectGrid(lattice=ions.cell.array, nr=[4, 4, 4])
    s = grid.s
    data = np.sin(2.0 * np.pi * s[0]) + np.cos(2.0 * np.pi * s[1])
    field = DirectField(grid, data=data)

    path = tmp_path / "data.xsf"
    write_xsf(path, ions=ions, data=field, units="angstrom")

    read_ions, read_field, _ = read_xsf(path, kind="all", units="angstrom")
    assert np.allclose(
        read_ions.to_ase().get_positions(), ions.to_ase().get_positions()
    )
    assert np.allclose(read_ions.to_ase().cell.array, ions.to_ase().cell.array)
    assert np.allclose(read_field, field, atol=1.0e-10)


def test_cube_roundtrip_with_data(tmp_path):
    ions = _ions_au()
    grid = DirectGrid(lattice=ions.cell.array, nr=[3, 3, 3])
    data = np.arange(grid.nnrR, dtype=float).reshape(grid.nr)
    field = DirectField(grid, data=data)

    path = tmp_path / "data.cube"
    write_cube(path, ions=ions, data=field)

    read_ions, read_field, _ = read_cube(path, kind="all")
    assert np.allclose(
        read_ions.to_ase().get_positions(), ions.to_ase().get_positions()
    )
    assert np.allclose(read_ions.to_ase().cell.array, ions.to_ase().cell.array)
    assert np.allclose(read_field, field)
