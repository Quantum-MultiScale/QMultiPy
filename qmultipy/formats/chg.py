from qmultipy.grid import DirectGrid
from qmultipy.field import DirectField
from qmultipy.ions import Ions
from qmultipy.constants import Units
from ase.calculators.vasp import VaspChargeDensity
import ase.io

def read_chg(infile, kind="all", full=False, **kwargs):
    if kind == "ions":
        ions = Ions.from_ase(ase.io.vasp.read_vasp(infile))
        values = ions
    else :
        obj = VaspChargeDensity(infile)
        ions = Ions.from_ase(obj.atoms[0])
        data = obj.chg[0]
        nr = data.shape
        grid = DirectGrid(lattice=ions.cell, nr=nr, full=full)
        fac = Units.Bohr**3
        if len(obj.chgdiff) > 0 :
            rank = 2
            total = obj.chg[0]
            diff = obj.chgdiff[0]
            data = [0.5*fac*(total + diff), 0.5*fac*(total - diff)]
        else :
            rank = 1
            data = data * fac
        plot = DirectField(grid=grid, data=data, rank=rank)
        values = (ions, plot, None)
    if kind == 'data' :
        return plot
    else :
        return values

def write_chg(filename, ions, data = None, format = None, **kwargs):
    fac = 1.0/Units.Bohr**3
    atoms = ions.to_ase()
    data = data*fac
    obj = VaspChargeDensity(filename=None)
    obj.atoms = [atoms]
    if data.rank > 1 :
        obj.chg = [data[0]+data[1]]
        obj.chgdiff = [data[0]-data[1]]
    else :
        obj.chg = [data]
    obj.write(filename, format=format)

def read_chgcar(infile, **kwargs):
    return read_chg(infile, **kwargs)

def write_chgcar(filename, ions, **kwargs):
    return write_chg(filename, ions, format='chgcar', **kwargs)
