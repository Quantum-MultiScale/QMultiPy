import numpy as np
import pytest

from qmultipy.field import DirectField
from qmultipy.grid import DirectGrid


@pytest.fixture
def grid():
    return DirectGrid(lattice=np.eye(3), nr=[8, 8, 8])


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


def test_interpolation_preserves_constant(grid):
    data = np.full(tuple(grid.nr), 3.7)
    field = DirectField(grid, data=data)
    interp = field.get_3dinterpolation([6, 6, 6])
    assert np.allclose(interp, 3.7, atol=1.0e-12)
