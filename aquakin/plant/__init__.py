"""Plant-wide simulation: compose kinetic reactors with non-reactive unit ops.

Provides the :class:`Plant` flowsheet that integrates a network of
:class:`Unit`-Protocol-conforming components (CSTRs, clarifiers, mixers,
splitters, controllers) under one monolithic Diffrax integration. The
plant graph supports recycles natively (recycle streams are functions of
current state, not future state, so no fixed-point iteration is needed).

BSM1 (Copp 2002 / Alex et al. 2008) is the first plant-wide validation
target — see :mod:`aquakin.plant.bsm.bsm1`.
"""

from aquakin.plant.aeration_system import (
    AerationDesignPoint,
    AerationSystem,
    blower_airflow_total,
    blower_energy,
    blower_power_kw,
    design_summary,
    required_airflow,
)
from aquakin.plant.balance import ComponentBalance, MassBalance, mass_balance
from aquakin.plant.control import PIController
from aquakin.plant.cstr import Aeration, CSTRUnit, oxygen_saturation
from aquakin.plant.disinfection import (
    ChlorineContactUnit,
    UVUnit,
    ct_log_removal,
    ct_value,
    t10_from_baffling,
    t10_from_rtd,
    uv_dose,
    uv_log_inactivation,
)
from aquakin.plant.dosing import DosingUnit, Reagent
from aquakin.plant.flow_setpoint import FlowParameterized, FlowSetpoint
from aquakin.plant.delay import HydraulicDelayUnit
from aquakin.plant.ifas import IFASUnit, MBBRUnit
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
from aquakin.plant.mbr import MBRUnit
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
from aquakin.plant.sbr import SBRPhase, SBRUnit
from aquakin.plant.separators import IdealThickener
from aquakin.plant.settling import (
    InterfaceSettling,
    LayeredSettling,
    SettlingModel,
)
from aquakin.plant.steady import PTCResult, ptc_forward, solve_steady_state
from aquakin.plant.storage import StorageTank
from aquakin.plant.streams import Stream, StreamSeries
from aquakin.plant.temperature import (
    AlgebraicTemperature,
    HeatBalanceTemperature,
    TemperatureModel,
)
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
from aquakin.plant.ghg import (
    CarbonFootprint,
    carbon_footprint,
    co2e_from_energy,
    methane_to_co2e,
    n2o_n_to_co2e,
    stripped_n2o,
)
from aquakin.plant.cost import CostFactors, OperatingCost, operating_cost
# Imported after the base units (they import the plant submodules above).
from aquakin.plant.a2o import (
    A2O_WARM_REACTOR_COMPOSITION,
    FerricDose,
    a2o_influent,
    a2o_warm_start,
    build_a2o,
)
# Imported last: the BSM evaluators pull in the bsm subpackage (build_bsm1 etc.),
# which imports the plant submodules above -- so the base units must load first.
from aquakin.plant.bsm.evaluation import (
    BSM1Evaluation,
    BSM2Evaluation,
    direct_n2o_emission,
    evaluate_bsm1,
    evaluate_bsm2,
)

__all__ = [
    "A2O_WARM_REACTOR_COMPOSITION",
    "ADM1DigesterUnit",
    "FerricDose",
    "ADM1toASM1",
    "ASM1toADM1",
    "ActivatedSludgeSizing",
    "BSM1Evaluation",
    "BSM2Evaluation",
    "Aeration",
    "AerationDesignPoint",
    "AerationSystem",
    "a2o_influent",
    "a2o_warm_start",
    "build_a2o",
    "blower_airflow_total",
    "blower_energy",
    "blower_power_kw",
    "design_summary",
    "required_airflow",
    "CSTRUnit",
    "DosingUnit",
    "Reagent",
    "UVUnit",
    "ChlorineContactUnit",
    "uv_dose",
    "uv_log_inactivation",
    "ct_value",
    "ct_log_removal",
    "t10_from_baffling",
    "t10_from_rtd",
    "IFASUnit",
    "MBBRUnit",
    "FlowParameterized",
    "FlowSetpoint",
    "ComponentBalance",
    "MassBalance",
    "mass_balance",
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
    "MBRUnit",
    "MixerUnit",
    "PIController",
    "ParameterLayout",
    "PiecewiseConstantSchedule",
    "Plant",
    "PlantCheck",
    "PlantSolution",
    "PrimaryClarifier",
    "PTCResult",
    "FlowContext",
    "SludgeMetrics",
    "SteadyStateResult",
    "ptc_forward",
    "solve_steady_state",
    "SBRPhase",
    "SBRUnit",
    "SettlingModel",
    "InterfaceSettling",
    "LayeredSettling",
    "SplitterUnit",
    "StatelessUnit",
    "StateTranslator",
    "StorageTank",
    "Stream",
    "StreamSeries",
    "TemperatureModel",
    "AlgebraicTemperature",
    "HeatBalanceTemperature",
    "Unit",
    "CarbonFootprint",
    "CostFactors",
    "OperatingCost",
    "aeration_energy",
    "carbon_footprint",
    "carbon_mass",
    "co2e_from_energy",
    "derived_BOD",
    "derived_COD",
    "derived_TKN",
    "derived_TSS",
    "direct_n2o_emission",
    "effluent_averages",
    "effluent_quality_index",
    "evaluate_bsm1",
    "evaluate_bsm2",
    "heating_energy",
    "methane_to_co2e",
    "mixing_energy",
    "n2o_n_to_co2e",
    "operating_cost",
    "operational_cost_index",
    "operational_cost_index_bsm2",
    "pumping_energy",
    "pumping_energy_bsm2",
    "size_activated_sludge",
    "sludge_metrics",
    "stripped_n2o",
]
