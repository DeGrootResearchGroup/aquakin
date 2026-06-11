"""Plant-wide simulation: compose kinetic reactors with non-reactive unit ops.

Provides the :class:`Plant` flowsheet that integrates a network of
:class:`Unit`-Protocol-conforming components (CSTRs, clarifiers, mixers,
splitters, controllers) under one monolithic Diffrax integration. The
plant graph supports recycles natively (recycle streams are functions of
current state, not future state, so no fixed-point iteration is needed).

BSM1 (Copp 2002 / Alex et al. 2008) is the first plant-wide validation
target — see :mod:`aquakin.plant.bsm.bsm1`.
"""

from aquakin.plant.control import PIController
from aquakin.plant.cstr import CSTRUnit
from aquakin.plant.digester import ADM1DigesterUnit
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.interfaces import ADM1toASM1, ASM1toADM1
from aquakin.plant.mixer import MixerUnit, SplitterUnit
from aquakin.plant.plant import (
    Connection,
    ParameterLayout,
    Plant,
    PlantSolution,
)
from aquakin.plant.primary_clarifier import PrimaryClarifier
from aquakin.plant.separators import IdealThickener
from aquakin.plant.streams import Stream, StreamSeries
from aquakin.plant.translators import IdentityTranslator, StateTranslator
from aquakin.plant.units import Unit

__all__ = [
    "ADM1DigesterUnit",
    "ADM1toASM1",
    "ASM1toADM1",
    "CSTRUnit",
    "Connection",
    "IdentityTranslator",
    "IdealThickener",
    "InfluentSeries",
    "MixerUnit",
    "PIController",
    "ParameterLayout",
    "Plant",
    "PlantSolution",
    "PrimaryClarifier",
    "SplitterUnit",
    "StateTranslator",
    "Stream",
    "StreamSeries",
    "Unit",
]
