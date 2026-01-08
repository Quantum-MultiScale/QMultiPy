import numpy as np
import pytest
from pyscf import gto

from qmultipy.field import DirectField
from qmultipy.grid import DirectGrid


@pytest.fixture
def simple_grid():
    # Create a simple cubic grid
    lattice = np.eye(3) * 6.0  # 10 Bohr cube
    nr = np.array([20, 20, 20])  # 20x20x20 grid points
    return DirectGrid(lattice=lattice, nr=nr)


@pytest.fixture
def simple_mol():
    # Create a simple H2 molecule
    mol = gto.M(
        atom='''
        He 3 3 3
        ''',
        basis='6-31g',
        unit='Bohr',
    )
    mol.build()
    # print("mol.atom_coords",mol.atom_coords())
    return mol


@pytest.fixture
def mol_grid(simple_mol):
    # Create a simple cubic grid
    from pyscf.dft.gen_grid import Grids

    grids = Grids(simple_mol)
    grids.level = 4
    grids.build()
    # np.savetxt("grids.txt", grids.coords)
    return grids.coords, grids.weights


@pytest.fixture
def gaussian_field_cartesian(simple_grid):
    # Create a Gaussian-shaped field centered in the box
    center = np.array([3.0, 3.0, 3.0])
    sigma = 0.7
    r2 = (
        (simple_grid.r[0] - center[0]) ** 2
        + (simple_grid.r[1] - center[1]) ** 2
        + (simple_grid.r[2] - center[2]) ** 2
    )
    data = np.exp(-r2 / (2 * sigma**2))
    return DirectField(grid=simple_grid, data=data)


@pytest.fixture
def gaussian_field_molecular(mol_grid):
    # Create a Gaussian-shaped field centered in the box
    center = np.array([3.0, 3.0, 3.0])
    sigma = 0.7
    r2 = (
        (mol_grid[0][:, 0] - center[0]) ** 2
        + (mol_grid[0][:, 1] - center[1]) ** 2
        + (mol_grid[0][:, 2] - center[2]) ** 2
    )
    data = np.exp(-r2 / (2 * sigma**2))
    return data


# def test_field_to_othergrid(gaussian_field, simple_grid):
#    field = gaussian_field
#    othergrid = simple_grid
#    otherfield = field.to_othergrid(othergrid)
#    assert otherfield is not None # just to check if it runs next is to get the coeffs from a calc and hardcode assert
#


def rmse(predictions, targets):
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    # print("rmse", rmse)
    return rmse


def test_field_to_molgrid(gaussian_field_cartesian, gaussian_field_molecular, mol_grid):
    field = gaussian_field_cartesian
    # field.spl_order = 3
    field.spl_coeffs = None
    othergrid = mol_grid[0]
    otherfield = field.to_molecular_grid(othergrid, fast=False)
    # np.savetxt("gaussian_field_cartesian.txt", field.flatten())
    # np.savetxt("otherfield.txt", otherfield)
    # np.savetxt("gaussian_field_molecular.txt", gaussian_field_molecular)
    assert rmse(otherfield, gaussian_field_molecular) < 5e-2


def test_field_to_othergrid_fast(
    gaussian_field_cartesian, gaussian_field_molecular, mol_grid
):
    field = gaussian_field_cartesian
    othergrid = mol_grid[0]
    otherfield = field.to_molecular_grid(othergrid, fast=True)
    # print("max delta", np.max(np.abs(otherfield - gaussian_field_molecular)))
    assert rmse(otherfield, gaussian_field_molecular) < 5e-2


def test_field_to_GTOs(gaussian_field_cartesian, simple_mol):
    field = gaussian_field_cartesian
    mol = simple_mol
    grid_level = 4
    gto = field.to_GTOs(mol, grid_level)
    assert (
        gto is not None
    )  # just to check if it runs next is to get the coeffs from a calc and hardcode assert


def test_field_to_mat_GTOs(gaussian_field_cartesian, simple_mol):
    field = gaussian_field_cartesian
    mol = simple_mol
    mat = field.to_mat_GTOs(mol)
    assert (
        mat is not None
    )  # just to check if it runs next is to get the matrix from a calc and hardcode assert
