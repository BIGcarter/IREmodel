import numpy as np

# Constants from feria.h
EPS = 1e-308
INF = 1e+308
PI = np.pi
DB_ERRATIO = 5e-15

C_LIGHT = 2.99792458e+10      # cm s^-1
K_BOLTZ = 1.3806488e-16       # cm^2 g s^-2 K^-1
GRAV = 6.67384e-8             # cm^3 g^-1 s^-2
H_PLANCK = 6.62606957e-27     # cm^2 g s^-1
MSUN = 1.989e+33              # g
CMPERM = 1e+2                 # cm
CMPERKM = 1e+5                # cm
CMPERAU = 1.49597871e+13      # cm
ASPERRAD = 180. * 3600. / PI
DEGPERRAD = 180. / PI
CMPERPC = CMPERAU * ASPERRAD  # cm

FITS_EXT = ".fits"
PARAMS_NAME = "parameter"
EQUINOX = "J2000"

# Axis indices
XAXIS = 0
RAXIS = 0
YAXIS = 1
TAXIS = 1
ZAXIS = 2
VAXIS = 2
NAXIS = 3

PAXIS_PV = 0
VAXIS_PV = 1
NAXIS_PV = 2

# Gas types
NO_GAS = 0
IS_IRE = 1
IS_KEP = 2

# Data indices
VX_VAL = 0
VY_VAL = 1
VZ_VAL = 2
N_VAL = 3
T_VAL = 4
NDATA = 5
