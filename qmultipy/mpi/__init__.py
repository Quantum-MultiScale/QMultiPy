import numpy as np
import sys
from .utils import mp, sprint, pmi
from .mpi import MP, SerialComm, MPIFile, PMI

import builtins
builtins.__dict__['sprint'] = sprint

# numpy array print without truncation
np.set_printoptions(threshold=sys.maxsize)
