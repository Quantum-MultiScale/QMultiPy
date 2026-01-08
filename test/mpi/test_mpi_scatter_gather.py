import numpy as np
import pytest

from qmultipy.field import DirectField
from qmultipy.grid import DirectGrid
from qmultipy.mpi import MP


def _require_mpi():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.size < 2:
        pytest.skip("MPI test requires at least 2 ranks")
    return comm


def test_grid_scatter_gather_roundtrip():
    comm = _require_mpi()
    mp = MP(comm=comm)
    grid = DirectGrid(lattice=np.eye(3), nr=[4, 4, 4], mp=mp)

    if comm.rank == 0:
        global_data = np.arange(grid.nnrR, dtype=float).reshape(grid.nrR)
    else:
        global_data = np.empty(grid.nrR, dtype=float)

    local = grid.scatter(global_data, root=0)
    assert local.shape == tuple(grid.nr)

    gathered = grid.gather(local, root=0)
    if comm.rank == 0:
        assert np.array_equal(gathered, global_data)


def test_field_gather_roundtrip():
    comm = _require_mpi()
    mp = MP(comm=comm)
    grid = DirectGrid(lattice=np.eye(3), nr=[4, 4, 4], mp=mp)

    if comm.rank == 0:
        global_data = np.arange(grid.nnrR, dtype=float).reshape(grid.nrR)
    else:
        global_data = np.empty(grid.nrR, dtype=float)

    local = grid.scatter(global_data, root=0)
    field = DirectField(grid, data=local)

    gathered = field.gather(root=0)
    if comm.rank == 0:
        assert np.array_equal(gathered, global_data)


def test_mpi_collectives_shape_metadata():
    comm = _require_mpi()
    mp = MP(comm=comm)
    grid = DirectGrid(lattice=np.eye(3), nr=[4, 4, 4], mp=mp)

    assert len(grid.slice_all) == comm.size
    assert len(grid.nr_all) == comm.size
    assert len(grid.offsets_all) == comm.size
