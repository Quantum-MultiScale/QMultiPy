import numpy as np
import pytest

from qmultipy.field import DirectField, Field, ReciprocalField
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


def test_get_cut_constant_field_1d(grid):
    data = np.full(tuple(grid.nr), 2.5)
    field = DirectField(grid, data=data)

    cut = field.get_cut(r0=[1.0, 0.0, 0.0], origin=[0.0, 0.0, 0.0], nr=12)
    assert np.allclose(cut, 2.5, atol=1.0e-12)
    assert np.array_equal(cut.grid.nr, np.array([12, 1, 1]))

    cut_centered = field.get_cut(r0=[1.0, 0.0, 0.0], center=[0.5, 0.0, 0.0], nr=8)
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
