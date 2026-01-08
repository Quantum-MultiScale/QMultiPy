from copy import deepcopy

import numpy as np
from ase import Atoms
from ase.atom import Atom
from ase.atoms import default
from ase.cell import Cell
from ase.symbols import symbols2numbers

from qmultipy.constants import Units


class IonsBase:

    ase_methods = [
        "new_array",
        "set_cell",
        "set_celldisp",
        "get_celldisp",
        "set_tags",
        "get_tags",
        "set_array",
        "set_initial_magnetic_moments",
        "get_initial_magnetic_moments",
        "set_initial_charges",
        "get_initial_charges",
        "get_scaled_positions",
        "get_chemical_formula",
        "get_cell",
        "get_chemical_symbols",
        "set_pbc",
        "set_positions",
        "get_positions",
        "has",
        "copy",
        "wrap",
    ]
    ase_attributes = [
        "init_options",
        "symbols",
        "positions",
        "numbers",
        "tags",
        "magmoms",
        "scaled_positions",
        "cell",
        "celldisp",
        "info",
        "velocities",
        "arrays",
        "pbc",
        "constraints",  # please do not use this
    ]

    def __getattribute__(self, name):
        if name in IonsBase.ase_attributes or name in IonsBase.ase_methods:
            attr = object.__getattribute__(self._atoms, name)
        else:
            attr = object.__getattribute__(self, name)
        return attr

    def __init__(self, atoms: Atoms):
        self.atoms = atoms

    @property
    def atoms(self) -> Atoms:
        at = self._atoms.copy()

        positions = self.get_positions() * Units.Bohr
        magmoms = self.get_initial_magnetic_moments() * (Units.A * Units.Bohr**2)
        cell = self.get_cell().array * Units.Bohr
        celldisp = self.get_celldisp() * Units.Bohr

        at.set_positions(positions)
        at.set_initial_magnetic_moments(magmoms)
        at.set_cell(cell)
        at.set_celldisp(celldisp)

        return at

    @atoms.setter
    def atoms(self, value: Atoms):
        self._atoms = value.copy()

        positions = self.get_positions() / Units.Bohr
        magmoms = self.get_initial_magnetic_moments() / (Units.A * Units.Bohr**2)
        cell = self.get_cell().array / Units.Bohr
        celldisp = self.get_celldisp() / Units.Bohr

        self.set_positions(positions)
        self.set_initial_magnetic_moments(magmoms)
        self.set_cell(cell)
        self.set_celldisp(celldisp)


class Ions(IonsBase):
    """Ions object based on `ase.Atoms <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_

    .. note::

        Only change the units of length, and others still keep the units of ASE.

             - positions : Bohr
             - cell : Bohr
             - celldisp : Bohr

    """

    def __init__(
        self,
        atoms=None,
        symbols=None,
        positions=None,
        numbers=None,
        tags=None,
        magmoms=None,
        charges=None,
        scaled_positions=None,
        cell=None,
        celldisp=None,
        info=None,
        pbc=True,
        units='au',
    ):
        if units not in ['au', 'ase']:
            raise ValueError("units must be either 'au' (Bohr) or 'ase' (Angstrom)")

        if atoms is not None:
            init_options = locals()
            for k in ["__class__", "self", "atoms", "pbc", "units"]:
                init_options.pop(k, None)
            if any([o is not None for o in init_options.values()]):
                raise TypeError(
                    'When initializing Ions from atoms object, please do not pass any other arguments'
                )
            super().__init__(atoms)
        else:
            if symbols is not None and numbers is not None:
                raise TypeError('Use only one of "symbols" and "numbers".')
            if symbols is not None:
                numbers = symbols2numbers(symbols)
            elif numbers is None:
                if positions is not None:
                    natoms = len(positions)
                elif scaled_positions is not None:
                    natoms = len(scaled_positions)
                else:
                    natoms = 0
                numbers = np.zeros(natoms, int)

            if cell is None:
                cell = np.zeros((3, 3))
            elif units == 'au':
                cell = np.array(cell) * Units.Bohr
            if celldisp is None:
                celldisp = np.zeros(shape=(3, 1))
            elif units == 'au':
                celldisp = np.array(celldisp) * Units.Bohr

            if positions is None:
                if scaled_positions is None:
                    positions = np.zeros((len(numbers), 3))
                else:
                    assert cell.rank == 3
                    positions = np.dot(scaled_positions, cell)
            else:
                if units == 'au':
                    positions = np.array(positions) * Units.Bohr
                if scaled_positions is not None:
                    raise TypeError(
                        'Use only one of "positions" and "scaled_positions".'
                    )

            if magmoms is not None:
                if units == 'au':
                    magmoms = np.array(magmoms) * (Units.A * Units.Bohr**2)

            init_options = locals()
            for k in ["__class__", "self", "symbols", "atoms", "units"]:
                init_options.pop(k, None)

            at = Atoms(**init_options)
            super().__init__(at)

    def to_ase(self):
        return self.atoms

    @staticmethod
    def from_ase(atoms: Atoms):
        return Ions(atoms)

    def get_ncharges(self):
        """Get total number of charges."""
        if not self.has("initial_charges"):
            raise AttributeError("Please call 'set_charges' before use 'charges'.")
        return self.arrays["initial_charges"].sum()

    def get_charges(self):
        """Get the atomic charges."""
        return self.get_initial_charges()

    def set_charges(self, charges=None):
        """Set the atomic charges."""
        if isinstance(charges, dict):
            values = []
            for s in self.symbols:
                if s not in charges:
                    raise AttributeError(f"{s} not in the charges")
                values.append(charges[s])
            charges = values
        elif isinstance(charges, (float, int)):
            charges = np.ones(self.nat) * charges
        self.set_initial_charges(charges=charges)

    @property
    def charges(self):
        """Get the atomic charges."""
        if not self.has("initial_charges"):
            raise AttributeError("Please call 'set_charges' before use 'charges'.")
        return self.arrays["initial_charges"]

    @charges.setter
    def charges(self, value):
        """Set the atomic charges."""
        if not self.has("initial_charges"):
            raise AttributeError("Please call 'set_charges' before use 'charges'.")
        self.arrays["initial_charges"][:] = value

    def strf(self, reciprocal_grid, iatom):
        """Returns the Structure Factor associated to i-th ion."""
        a = np.exp(
            -1j * np.einsum("lijk,l->ijk", reciprocal_grid.g, self.positions[iatom])
        )
        return a

    def istrf(self, reciprocal_grid, iatom):
        """Returns the Structure-Factor-like property associated to i-th ion."""
        a = np.exp(
            1j * np.einsum("lijk,l->ijk", reciprocal_grid.g, self.positions[iatom])
        )
        return a

    @property
    def symbols_uniq(self):
        """Unique symbols of ions"""
        return np.sort(np.unique(self.symbols))

    @property
    def nat(self):
        """Number of atoms"""
        return len(self._atoms)

    @property
    def zval(self):
        """Valance charge (atomic charge) of each atomic type"""
        zval = dict.fromkeys(self.symbols_uniq, 0)
        symbols = self.get_chemical_symbols()
        try:
            self.charges[0]
        except Exception:
            return zval

        for k in zval:
            for i in range(self.nat):
                if symbols[i] == k:
                    zval[k] = self.charges[i]
                    break
        return zval

    def repeat(self, *args, **kwargs):
        ions = deepcopy(self)
        ions._atoms = ions._atoms.repeat(*args, **kwargs)
        return ions
