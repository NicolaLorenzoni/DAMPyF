"""
    unit-conversions
"""

import numpy as np

SPEED_LIGHT_CM_PER_SEC = 2.99792458e10
EV_TO_INV_CM = 8065.543730730113
BOLTZMANN_CONST_EV_PER_K = 8.617333262145e-5


# =============================================================================
# Spectroscopic-convention constants
# =============================================================================

BOLTZMANN_CONST_INV_CM_PER_K = BOLTZMANN_CONST_EV_PER_K * EV_TO_INV_CM
FEMTOSECOND_TO_SPECTROSCOPIC_TIME = 2.0 * np.pi * SPEED_LIGHT_CM_PER_SEC * 1.0e-15
