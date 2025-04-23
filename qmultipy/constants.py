import os
import sys
from ase.units import create_units
Units = create_units('2018')

try:
    import pyfftw
    FFTLIB = "pyfftw"
except Exception:
    FFTLIB = "numpy"

def conv2conv(conv, base = None):
    if base is None : base = list(conv.keys())[0]
    conv[base][base] = 1.0
    ref = conv[base]
    for key in ref :
        if key == base : continue
        conv[key] = {}
        for key2 in ref :
            conv[key][key2] = ref[key2]/ref[key]
    return conv


LEN_CONV={"Angstrom" : {"Bohr": 1.0/Units.Bohr, "nm": 1.0e-1, "m": 1.0e-10}}
LEN_CONV = conv2conv(LEN_CONV)

ENERGY_CONV= {"Hartree": {"eV": Units.Ha}}
ENERGY_CONV = conv2conv(ENERGY_CONV)

FORCE_CONV = {"Ha/Bohr": {"eV/A" : Units.Ha/Units.Bohr}}
FORCE_CONV = conv2conv(FORCE_CONV)

STRESS_CONV = {"eV/A3" : {"GPa": 1.0/Units.GPa, "Ha/Bohr3" : Units.Bohr ** 3 / Units.Ha}}
STRESS_CONV = conv2conv(STRESS_CONV)

TIME_CONV = {"au" : {'s' : Units.AUT/Units.s, 'fs' : Units.AUT/Units.fs}}
TIME_CONV = conv2conv(TIME_CONV)

SPEED_OF_LIGHT = 1.0/Units.alpha
C_TF = 2.87123400018819181594
TKF0 = 6.18733545256027186194
CBRT_TWO = 1.25992104989487316477
ZERO = 1E-30

environ = {} # You can change it anytime you want
environ['STDOUT'] = sys.stdout # file descriptor of sprint
environ['LOGLEVEL'] = int(os.environ.get('qmultipy_LOGLEVEL', 2)) # The level of sprint
"""
    0 : all
    1 : debug
    2 : info
    3 : warning
    4 : error
"""
environ['FFTLIB'] = os.environ.get('qmultipy_FFTLIB', FFTLIB)
