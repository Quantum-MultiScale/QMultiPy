import pytest
import numpy as np
from qmultipy.grid import DirectGrid, ReciprocalGrid, RadialGrid, BaseGrid, Grid
from scipy.interpolate import splrep, splev
from ase.cell import Cell


@pytest.fixture
def lattice():
    return Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

@pytest.fixture
def radial_data():
    r = np.linspace(0, 10, 100)
    v = np.exp(-r)
    return r, v

def test_direct_grid_init(lattice):
    grid = DirectGrid(lattice=lattice, nr=[4, 4, 4])
    assert isinstance(grid, DirectGrid)
    assert grid.direct is True
    assert grid.full is True 

    # Check nrR, nnrR
    assert np.array_equal(grid.nrR, np.array([4,4,4]))
    assert grid.nnrR == 4*4*4

    # Check nr, nnr (testing for serial now)
    assert np.array_equal(grid.nr, np.array([4,4,4]))
    assert grid.nnr == 4*4*4

    # Check nrG, nnrG
    assert np.array_equal(grid.nrG, np.array([4,4,4]))
    assert grid.nnrG == 4*4*4
    
    #Check volume, dV, and spacings
    assert grid.volume == pytest.approx(1.0)
    expected_dV = 1.0/64.0
    assert grid.dV == pytest.approx(expected_dV)
    expected_sp = grid.cell.lengths()/grid.nrR
    assert np.allclose(grid.spacings, expected_sp)

    #Check lattice property and origins
    assert np.allclose(grid.lattice, Cell(lattice).array)
    assert np.array_equal(grid.origin, np.zeros(3))

    #Check cases where full=False and cplx=False 
    grid1 = DirectGrid(lattice=lattice, nr=[4, 4, 4], full=False, cplx=False)
    assert grid1.full is False
    assert np.array_equal(grid1.nrG, np.array([4,4,3]))
    assert grid1.nnrG == 4*4*3
   
    #Check cases where full=False and cplx=True (forces full=True)
    grid2 = DirectGrid(lattice=lattice, nr=[4, 4, 4], full=False, cplx=True)
    assert grid2.full is True
    assert np.array_equal(grid2.nrG, np.array([4,4,4]))
    assert grid2.nnrG == 4*4*4
  
def test_direct_grid_eq(lattice):
    grid1 = DirectGrid(lattice=lattice, nr=[10, 10, 10])
    grid2 = DirectGrid(lattice=lattice, nr=[10, 10, 10])
    grid3 = DirectGrid(lattice=lattice, nr=[5, 5, 5])

    assert grid1 == grid2
    assert grid1 != grid3

def test_direct_grid_tile(lattice):
    grid = DirectGrid(lattice=lattice, nr=[2, 2, 2])
    tiled_grid = grid.tile(reps=2)
    assert np.all(tiled_grid.nr == [4, 2, 2])
    assert np.allclose(tiled_grid.lattice, np.diag([2, 1, 1]))

    tiled_grid = grid.tile(reps=[1, 2, 3])
    assert np.all(tiled_grid.nr == [2, 4, 6])
    assert np.allclose(tiled_grid.lattice, np.diag([1, 2, 3]))


def test_direct_grid_repeat(lattice):
    grid = DirectGrid(lattice=lattice, nr=[2, 2, 2])
    grid.rank = 1 # The code is a bit buggy here. 
    repeated_grid = grid.repeat(2)
    assert np.all(repeated_grid.nr == [4, 4, 4])
    assert np.allclose(repeated_grid.lattice, lattice * 2)

def test_grid_create(lattice):
    grid = DirectGrid(lattice=lattice, nr=[10, 10, 10])
    new_grid = grid.create()
    assert isinstance(new_grid, DirectGrid)
    assert np.allclose(new_grid.lattice, grid.lattice)
    assert np.all(new_grid.nr == grid.nr)

    new_lattice = np.diag([2, 2, 2])
    new_grid = grid.create(lattice=new_lattice)
    assert np.allclose(new_grid.lattice, new_lattice)

def test_grid_local_slice(lattice):
    grid = DirectGrid(lattice=lattice, nr=[10, 10, 10])
    s = grid.slice
    assert grid.slice_all == s
    assert np.array_equal(grid.nr_all, grid.nr)
    assert np.array_equal(grid.offsets_all, grid.offsets)
    
def test_grid_gather_scatter(lattice):
    grid = DirectGrid(lattice=lattice, nr=[2, 3, 4])
    data = np.arange(np.prod(grid.nrR)).reshape(grid.nrR)
    gathered = grid.gather(data)
    scattered = grid.scatter(data)
    assert np.array_equal(gathered, data)
    assert np.array_equal(scattered, data)

def test_ecut_setter_and_getter(lattice):
    grid = DirectGrid(lattice=lattice, nr=[10, 10, 10])
    grid.ecut = 5.0
    assert grid.ecut == pytest.approx(5.0)

    rec_grid = ReciprocalGrid(lattice=lattice, nr=[10, 10, 10])
    expected = rec_grid.get_direct().guess_ecut()
    assert rec_grid.ecut == pytest.approx(expected)

    rec_grid.ecut = 8.0
    assert rec_grid.ecut == pytest.approx(8.0)

def test_direct_grid_calc_points(lattice):
    grid = DirectGrid(lattice=lattice, nr=[3, 3, 3])
    s = grid.s
    assert s.shape == (3, 3, 3, 3)
    assert np.all(s >= 0) and np.all(s <= 1)

    r = grid.r
    assert r.shape == (3, 3, 3, 3)
    assert np.allclose(r, s)

    i,j,k = (1, 1, 1)
    assert s[0,i,j,k] == pytest.approx(1/3)  
    assert s[1,i,j,k] == pytest.approx(1/3)  
    assert s[2,i,j,k] == pytest.approx(1/3) 


def test_direct_grid_get_reciprocal(lattice):
    grid = DirectGrid(lattice=lattice, nr=[10, 10, 10])
    reciprocal_grid = grid.get_reciprocal()
    assert isinstance(reciprocal_grid, ReciprocalGrid)
    assert not reciprocal_grid.direct
    assert np.array_equal(reciprocal_grid.nrR, grid.nrR)

    expected_lat = 2.0 * np.pi * np.linalg.inv(grid.lattice).T
    assert np.allclose(reciprocal_grid.lattice, expected_lat)

    direct2 = reciprocal_grid.get_direct()
    assert isinstance(direct2, DirectGrid)
    assert np.allclose(direct2.lattice, grid.lattice)

def test_direct_grid_get_Rtable(lattice):
    grid = DirectGrid(lattice=lattice, nr=[3, 3, 3])
    rcut = 3.0
 
    Rtab = grid.get_Rtable(rcut=rcut)
    assert isinstance(Rtab, dict)
    assert "Nmax" in Rtab and "table" in Rtab

    expected_Nmax_val = int(np.ceil(rcut * grid.nr[0]) + 1)
    expected_Nmax = np.array([expected_Nmax_val] * 3, dtype=np.int32)
    assert Rtab["Nmax"].dtype == np.int32
    assert np.array_equal(Rtab["Nmax"], expected_Nmax)

    length = 2 * (expected_Nmax_val - 1) + 1
    table = Rtab["table"]
    assert table.shape == (length, length, length)

    center_idx = (expected_Nmax_val - 1,) * 3
    assert table[center_idx] == pytest.approx(0.0)

    assert grid.get_Rtable(rcut=rcut) is Rtab

def test_direct_grid_get_array_mask(lattice):
    grid = DirectGrid(lattice=lattice, nr=[4, 4, 4])
    xyz = np.vstack([np.arange(4), np.arange(4), np.arange(4)])
    mask = grid.get_array_mask(xyz.copy())
    assert mask == slice(None)

    data = np.array([10, 20, 30, 40])
    assert np.array_equal(data[mask], data)

def test_reciprocal_grid_init(lattice):
    grid = ReciprocalGrid(lattice=lattice, nr=[10, 10, 10])
    assert isinstance(grid, ReciprocalGrid)
    assert not grid.direct


def test_reciprocal_grid_eq(lattice):
    grid1 = ReciprocalGrid(lattice=lattice, nr=[10, 10, 10])
    grid2 = ReciprocalGrid(lattice=lattice, nr=[10, 10, 10])
    grid3 = ReciprocalGrid(lattice=lattice, nr=[5, 5, 5])

    assert grid1 == grid2
    assert grid1 != grid3

def test_reciprocal_grid_g_properties(lattice):
    rec_grid = ReciprocalGrid(lattice=lattice, nr=[10, 10, 10])
    g = rec_grid.g
    assert g.shape == (3, 10, 10, 10)
    assert np.allclose(g[0, 0, 0, 0], 0.0) 

    gg = rec_grid.gg
    assert gg.shape == (10, 10, 10)
    assert gg[0, 0, 0] == 0.0 

    q = rec_grid.q
    assert q.shape == gg.shape
    assert np.allclose(q, np.sqrt(gg))
    
    invgg = rec_grid.invgg
    assert invgg.shape == gg.shape
    assert invgg[0, 0, 0] == 0.0
    idx = (1, 0, 0)
    assert invgg[idx] == pytest.approx(1.0 / gg[idx])

    invq = rec_grid.invq
    assert invq.shape == q.shape
    assert invq[0, 0, 0] == 0.0
    idx = (0, 1, 0)
    assert invq[idx] == pytest.approx(1.0 / q[idx])

def test_reciprocal_grid_get_direct(lattice):
    grid = ReciprocalGrid(lattice=lattice, nr=[5, 5, 5])
    direct_grid = grid.get_direct()
    assert isinstance(direct_grid, DirectGrid)
    assert direct_grid.direct
    assert np.array_equal(direct_grid.nrR, grid.nrR)

    expected_lat = 2.0 * np.pi * np.linalg.inv(grid.lattice).T
    assert np.allclose(direct_grid.lattice, expected_lat)

    rec2 = direct_grid.get_reciprocal()
    assert isinstance(rec2, ReciprocalGrid)
    assert np.allclose(rec2.lattice, grid.lattice)

def test_reciprocal_grid_mask(lattice):
    grid = ReciprocalGrid(lattice=lattice, nr=[2, 2, 2], full=True)
    mask_serial = grid.mask_serial
    
    expected_mask = np.array([
        [[ True, False],
         [False, False]],
        [[ True, False],
         [False, False]]
    ], dtype=bool)

    assert np.array_equal(mask_serial, expected_mask)

    grid1 = ReciprocalGrid(lattice=lattice, nr=[2, 2, 2], full=True)
    mask = grid1.mask
    assert np.array_equal(mask, expected_mask)

def test_reciprocal_grid_gF(lattice):
    grid = ReciprocalGrid(lattice=lattice, nr=[4, 4, 4], ecut=1.0)
    gF = grid.gF
    ggF = grid.ggF
    expected_ggF = np.einsum("lijk,lijk->ijk", gF, gF)
    assert np.allclose(ggF, expected_ggF)

    assert grid.g2max == pytest.approx(2.0 * grid.ecut)

def test_radial_grid_init_setters(radial_data):
    r_in, v_in = radial_data
    grid = RadialGrid(r=r_in, v=v_in, direct=False)
    
    assert isinstance(grid, RadialGrid)
    assert np.array_equal(grid.r, r_in)
    assert np.array_equal(grid.v, v_in)
    assert grid._v_interp is None
    assert not grid.direct
    assert grid.vr is None
    
    new_r = r_in * 2.0
    grid.r = new_r
    assert np.array_equal(grid.r, new_r)

    new_v = v_in / 2.0
    grid.v = new_v
    assert np.array_equal(grid.v, new_v)
    
def test_radial_grid_interpolation(radial_data):
    r_in, v_in = radial_data
    grid = RadialGrid(r=r_in, v=v_in)

    test_r = np.array([0.0, 1.0, 2.0])
    expected_v = np.exp(-test_r)
    interpolated_v = splev(test_r, grid.v_interp)
    assert np.allclose(interpolated_v, expected_v)

def test_radial_grid_to_3d(radial_data):
    r_in, v_in = radial_data
    grid = RadialGrid(r=r_in, v=v_in)
    
    dist = np.array([[0.0, 1.0, 2.0],
                    [1.0, np.sqrt(2), np.sqrt(3)],
                    [2.0, np.sqrt(5), np.sqrt(8)]])


    v_3d = grid.to_3d_grid(dist)

    assert v_3d.shape == dist.shape
    assert np.allclose(v_3d[0, 0], 1.0) 
    assert np.allclose(v_3d[0, 1], np.exp(-1.0))

@pytest.mark.parametrize("method", ["simpson", "trapezoid"])
def test_ft_direct_and_indirect(radial_data, method):
    r_in, v_in = radial_data
    grid = RadialGrid(r=r_in, v=v_in)

    x = np.linspace(0.1, 1.0, 5)
    y_direct = grid.ft(x, method=method, direct=True, vr=False)
    assert y_direct.shape == x.shape
    
    grid2 = RadialGrid(r=r_in, v=v_in, direct=False, vr=False)
    y_ind = grid2.ft(x, method=method)
    assert not np.allclose(y_direct, y_ind)
    
    grid3 = RadialGrid(r=r_in, v=v_in, direct=True, vr=True)
    y_vr = grid3.ft(x, method=method)
    assert not np.allclose(y_vr, y_direct)

def test_grid(lattice):
    d = Grid(lattice, nr=[2,2,2], direct=True)
    assert isinstance(d, DirectGrid)
    r = Grid(lattice, nr=[4,4,4], direct=False)
    assert isinstance(r, ReciprocalGrid)

