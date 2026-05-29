"""IWA Benchmark Simulation Models (BSM family).

BSM1 is implemented in :mod:`bsm1`. BSM2 will follow once ADM1 lands.
"""

from aquakin.plant.bsm.bsm1 import build_bsm1
from aquakin.plant.influent import load_bsm1_influent

__all__ = ["build_bsm1", "load_bsm1_influent"]
