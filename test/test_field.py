import numpy as np
import pytest
from pyscf import gto
from qmultipy.field import DirectField
from qmultipy.grid import DirectGrid

@pytest.fixture
def simple_grid():
    # Create a simple cubic grid
    lattice = np.eye(3) * 10.0  # 10 Bohr cube
    nr = np.array([20, 20, 20])  # 20x20x20 grid points
    return DirectGrid(lattice=lattice, nr=nr)

@pytest.fixture
def simple_mol():
    # Create a simple H2 molecule
    mol = gto.M(
        atom='''
        H 0 0 0
        H 0 0 1
        ''',
        basis='6-31g',
        unit='Bohr'
    )
    mol.build()
    return mol

@pytest.fixture
def mol_grid(simple_mol):
    # Create a simple cubic grid
    from pyscf.dft.gen_grid import Grids
    grids = Grids(simple_mol)
    grids.level = 4
    grids.build()
    return grids.coords, grids.weights
    
@pytest.fixture
def gaussian_field(simple_grid):
    # Create a Gaussian-shaped field centered in the box
    center = np.array([5.0, 5.0, 5.0])
    sigma = 1.0
    r2 = np.einsum('ijkl, i->jkl',simple_grid.r,center)**2
    data = np.exp(-r2/(2*sigma**2))
    return DirectField(grid=simple_grid, data=data)

#def test_field_to_othergrid(gaussian_field, simple_grid):
#    field = gaussian_field
#    othergrid = simple_grid
#    otherfield = field.to_othergrid(othergrid)
#    assert otherfield is not None # just to check if it runs next is to get the coeffs from a calc and hardcode assert
#

def test_field_to_molgrid(gaussian_field, mol_grid):
    field = gaussian_field
    othergrid = mol_grid[0]
    field = field.to_molecular_grid(othergrid, fast=False)
    assert field is not None # just to check if it runs next is to get the coeffs from a calc and hardcode assert

def test_field_to_othergrid_fast(gaussian_field, mol_grid):
    field = gaussian_field
    othergrid = mol_grid[0]
    otherfield = field.to_molecular_grid(othergrid, fast=True)
    assert otherfield is not None # just to check if it runs next is to get the coeffs from a calc and hardcode assert

def test_field_to_GTOs(gaussian_field, simple_mol):
    field = gaussian_field
    mol = simple_mol
    grid_level = 4
    gto = field.to_GTOs(mol, grid_level)
    assert gto is not None # just to check if it runs next is to get the coeffs from a calc and hardcode assert

def test_field_to_mat_GTOs(gaussian_field, simple_mol):
    field = gaussian_field
    mol = simple_mol
    mat = field.to_mat_GTOs(mol)
    assert mat is not None # just to check if it runs next is to get the matrix from a calc and hardcode assert

