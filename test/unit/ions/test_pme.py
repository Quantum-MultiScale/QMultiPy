import numpy as np
from ase.build import bulk

from qmultipy.ewald import Ewald
from qmultipy.grid import Grid
from qmultipy.ions import Ions


def test_ewald_pme():
    print("*" * 50)
    print("Testing particle mesh Ewald method")
    atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
    ions = Ions.from_ase(atoms)
    ions.set_charges(3.0)
    grid = Grid(ions.cell, ecut=25)
    print('grid', grid.nr)
    ewald_ = Ewald(grid=grid, ions=ions)
    ewald_pme = Ewald(grid=grid, ions=ions, PME=True)

    print('Ewald energy', ewald_.energy - ewald_pme.energy)
    assert np.allclose(ewald_.energy, ewald_pme.energy, atol=1.0e-5)
    print('Ewald forces', ewald_.forces[0] - ewald_pme.forces[0])
    assert np.allclose(ewald_.forces, ewald_pme.forces, atol=1.0e-5)
    print('Ewald stress', ewald_.stress[0] - ewald_pme.stress[0])
    assert np.allclose(ewald_.stress, ewald_pme.stress, atol=1.0e-5)
