import numpy as np
import pytest

from qmultipy.field import DirectField, Field, ReciprocalField
from qmultipy.grid import DirectGrid


@pytest.fixture
def grid():
    return DirectGrid(lattice=np.eye(3), nr=[8, 8, 8])


@pytest.fixture
def simple_grid():
    lattice = np.eye(3) * 6.0
    nr = np.array([20, 20, 20])
    return DirectGrid(lattice=lattice, nr=nr)


@pytest.fixture
def simple_mol():
    pytest.importorskip("pyscf")
    from pyscf import gto

    mol = gto.M(
        atom="""
        He 3 3 3
        """,
        basis="6-31g",
        unit="Bohr",
    )
    mol.build()
    return mol


@pytest.fixture
def mol_grid(simple_mol):
    from pyscf.dft.gen_grid import Grids

    grids = Grids(simple_mol)
    grids.level = 4
    grids.build()
    return grids.coords, grids.weights


@pytest.fixture
def gaussian_field_cartesian(simple_grid):
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
    center = np.array([3.0, 3.0, 3.0])
    sigma = 0.7
    r2 = (
        (mol_grid[0][:, 0] - center[0]) ** 2
        + (mol_grid[0][:, 1] - center[1]) ** 2
        + (mol_grid[0][:, 2] - center[2]) ** 2
    )
    return np.exp(-r2 / (2 * sigma**2))


def _rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def _scalar_field(grid):
    s = grid.s
    data = (
        np.sin(2.0 * np.pi * s[0])
        + np.cos(2.0 * np.pi * s[1])
        + np.sin(2.0 * np.pi * s[2])
    )
    return DirectField(grid, data=data), s


def test_fft_ifft_roundtrip(grid):
    rng = np.random.default_rng(0)
    data = rng.random(tuple(grid.nr))
    field = DirectField(grid, data=data)
    back = field.fft().ifft(check_real=True, force_real=True)
    assert np.allclose(back, data, atol=1.0e-10)


def test_standard_gradient_matches_analytic(grid):
    field, s = _scalar_field(grid)
    grad = field.gradient(flag="standard", force_real=True)
    expected = np.zeros((3, *grid.nr))
    expected[0] = 2.0 * np.pi * np.cos(2.0 * np.pi * s[0])
    expected[1] = -2.0 * np.pi * np.sin(2.0 * np.pi * s[1])
    expected[2] = 2.0 * np.pi * np.cos(2.0 * np.pi * s[2])
    assert np.allclose(grad, expected, atol=1.0e-8)


def test_laplacian_matches_analytic(grid):
    field, s = _scalar_field(grid)
    lap = field.laplacian(force_real=True, sigma=0.0)
    expected = -((2.0 * np.pi) ** 2) * (
        np.sin(2.0 * np.pi * s[0])
        + np.cos(2.0 * np.pi * s[1])
        + np.sin(2.0 * np.pi * s[2])
    )
    assert np.allclose(lap, expected, atol=1.0e-8)


def test_divergence_of_vector_field(grid):
    s = grid.s
    data = np.zeros((3, *grid.nr))
    data[0] = np.sin(2.0 * np.pi * s[0])
    data[1] = np.sin(2.0 * np.pi * s[1])
    data[2] = np.sin(2.0 * np.pi * s[2])
    field = DirectField(grid, rank=3, data=data)
    div = field.divergence(flag="standard", force_real=True)
    expected = 2.0 * np.pi * (
        np.cos(2.0 * np.pi * s[0])
        + np.cos(2.0 * np.pi * s[1])
        + np.cos(2.0 * np.pi * s[2])
    )
    assert np.allclose(div, expected, atol=1.0e-8)


def test_get_value_at_points_preserves_constant(grid):
    data = np.full(tuple(grid.nr), 3.7)
    field = DirectField(grid, data=data)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.5, 0.75],
            [0.999, 0.1, 0.2],
        ]
    )
    values = field.get_value_at_points(points.copy())
    assert np.allclose(values, 3.7, atol=1.0e-12)


def test_get_cut_constant_field_1d(grid):
    data = np.full(tuple(grid.nr), 2.5)
    field = DirectField(grid, data=data)

    cut = field.get_cut(r0=[1.0, 0.0, 0.0], origin=[0.0, 0.0, 0.0], nr=12)
    assert np.allclose(cut, 2.5, atol=1.0e-12)
    assert np.array_equal(cut.grid.nr, np.array([12, 1, 1]))

    cut_centered = field.get_cut(
        r0=[1.0, 0.0, 0.0], center=[0.5, 0.0, 0.0], origin=None, nr=8
    )
    assert np.allclose(cut_centered, 2.5, atol=1.0e-12)


def test_para_current_plane_wave(grid):
    phase = 2.0 * np.pi * 0.5 * grid.s[0]
    data = np.exp(1j * phase)
    field = DirectField(grid, data=data, cplx=True)
    current = field.para_current(sigma=0.0)
    assert np.allclose(current[1:], 0.0, atol=1.0e-10)
    assert current[0].real == pytest.approx(0.40823666, abs=1.0e-6)
    assert current[0].imag == pytest.approx(2.64184126, abs=1.0e-6)


def test_field_factory_direct_and_reciprocal(grid):
    direct_field = Field(grid, data=np.zeros(tuple(grid.nr)), direct=True)
    assert isinstance(direct_field, DirectField)

    reciprocal_field = Field(grid, data=np.zeros(tuple(grid.nr)), direct=False)
    assert isinstance(reciprocal_field, ReciprocalField)


def test_field_to_molgrid(gaussian_field_cartesian, gaussian_field_molecular, mol_grid):
    field = gaussian_field_cartesian
    field.spl_coeffs = None
    otherfield = field.to_molecular_grid(mol_grid[0], fast=False)
    assert _rmse(otherfield, gaussian_field_molecular) < 5e-2


def test_field_to_othergrid_fast(
    gaussian_field_cartesian, gaussian_field_molecular, mol_grid
):
    field = gaussian_field_cartesian
    otherfield = field.to_molecular_grid(mol_grid[0], fast=True)
    assert _rmse(otherfield, gaussian_field_molecular) < 5e-2


def test_field_to_GTOs(gaussian_field_cartesian, simple_mol):
    field = gaussian_field_cartesian
    gto = field.to_GTOs(simple_mol, grid_level=4)
    assert gto is not None


def test_field_to_mat_GTOs(gaussian_field_cartesian, simple_mol):
    field = gaussian_field_cartesian
    mat = field.to_mat_GTOs(simple_mol, grid_level=4)
    assert mat is not None
