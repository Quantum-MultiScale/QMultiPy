#!/usr/bin/env python3
import unittest

import numpy as np
import pytest
from common import qmultipy_data_path


class Test(unittest.TestCase):
    def test_io_ase(self):
        from ase.io import read as ase_read

        from qmultipy.constants import Units
        from qmultipy.io import read as qmultipy_read

        qmultipy_ions = qmultipy_read(qmultipy_data_path / 'water.xyz', driver='ase')
        ase_atoms = ase_read(qmultipy_data_path / 'water.xyz')
        self.assertTrue(
            np.allclose(
                qmultipy_ions.get_positions() * Units.Bohr,
                ase_atoms.get_positions(),
                rtol=1.0e-15,
            )
        )


if __name__ == "__main__":
    unittest.main()
