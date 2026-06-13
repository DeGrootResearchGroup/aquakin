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
from aquakin.plant.cstr import Aeration, CSTRUnit, oxygen_saturation
from aquakin.plant.delay import HydraulicDelayUnit
from aquakin.plant.design import (
    ActivatedSludgeSizing,
    SludgeMetrics,
    size_activated_sludge,
    sludge_metrics,
)
from aquakin.plant.digester import ADM1DigesterUnit
from aquakin.plant.characterize import (
    InfluentFractions,
    characterize_influent,
    fractionate,
)
from aquakin.plant.influent import InfluentSeries, read_influent_csv
from aquakin.plant.interfaces import ADM1toASM1, ASM1toADM1
from aquakin.plant.mixer import MixerUnit, SplitterUnit
from aquakin.plant.plant import (
    Connection,
    ParameterLayout,
    Plant,
    PlantCheck,
    PlantSolution,
    SteadyStateResult,
)
from aquakin.plant.primary_clarifier import PrimaryClarifier
from aquakin.plant.schedule import PiecewiseConstantSchedule
from aquakin.plant.separators import IdealThickener
from aquakin.plant.storage import StorageTank
from aquakin.plant.streams import Stream, StreamSeries
from aquakin.plant.translators import IdentityTranslator, StateTranslator
from aquakin.plant.units import FlowContext, StatelessUnit, Unit
from aquakin.plant.metrics import (
    aeration_energy,
    carbon_mass,
    derived_BOD,
    derived_COD,
    derived_TKN,
    derived_TSS,
    effluent_averages,
    effluent_quality_index,
    heating_energy,
    mixing_energy,
    operational_cost_index,
    operational_cost_index_bsm2,
    pumping_energy,
    pumping_energy_bsm2,
)
# Imported last: the BSM evaluators pull in the bsm subpackage (build_bsm1 etc.),
# which imports the plant submodules above -- so the base units must load first.
from aquakin.plant.bsm.evaluation import (
    BSM1Evaluation,
    BSM2Evaluation,
    evaluate_bsm1,
    evaluate_bsm2,
)

__all__ = [
    "ADM1DigesterUnit",
    "ADM1toASM1",
    "ASM1toADM1",
    "ActivatedSludgeSizing",
    "BSM1Evaluation",
    "BSM2Evaluation",
    "Aeration",
    "CSTRUnit",
    "Connection",
    "oxygen_saturation",
    "HydraulicDelayUnit",
    "IdentityTranslator",
    "IdealThickener",
    "InfluentFractions",
    "InfluentSeries",
    "characterize_influent",
    "fractionate",
    "read_influent_csv",
    "MixerUnit",
    "PIController",
    "ParameterLayout",
    "PiecewiseConstantSchedule",
    "Plant",
    "PlantCheck",
    "PlantSolution",
    "PrimaryClarifier",
    "FlowContext",
    "SludgeMetrics",
    "SteadyStateResult",
    "SplitterUnit",
    "StatelessUnit",
    "StateTranslator",
    "StorageTank",
    "Stream",
    "StreamSeries",
    "Unit",
    "aeration_energy",
    "carbon_mass",
    "derived_BOD",
    "derived_COD",
    "derived_TKN",
    "derived_TSS",
    "effluent_averages",
    "effluent_quality_index",
    "evaluate_bsm1",
    "evaluate_bsm2",
    "heating_energy",
    "mixing_energy",
    "operational_cost_index",
    "operational_cost_index_bsm2",
    "pumping_energy",
    "pumping_energy_bsm2",
    "size_activated_sludge",
    "sludge_metrics",
]
