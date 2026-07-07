"""Plant-wide simulation: compose kinetic reactors with non-reactive unit ops.

Provides the :class:`Plant` flowsheet that integrates a model of
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
from aquakin.plant.calibrate import PlantObservable, calibrate_plant
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
from aquakin.plant.errors import (
    NoDigesterError,
    UnknownPortError,
    UnknownUnitError,
    WiringError,
)
from aquakin.plant.characterize import (
    InfluentFractions,
    characterize_influent,
    fractionate,
)
from aquakin.plant.influent import InfluentSeries, read_influent_csv
from aquakin.plant.interfaces import ADM1toASM1, ASM1toADM1
from aquakin.plant.mbr import MBRUnit
from aquakin.plant.mixer import (
    MixerUnit,
    RatioSplitter,
    SetpointSplitter,
    ThresholdSplitter,
)
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
    "ADM1toASM1",
    "ASM1toADM1",
    "ActivatedSludgeSizing",
    "Aeration",
    "AerationDesignPoint",
    "AerationSystem",
    "AlgebraicTemperature",
    "BSM1Evaluation",
    "BSM2Evaluation",
    "CSTRUnit",
    "CarbonFootprint",
    "ChlorineContactUnit",
    "ComponentBalance",
    "Connection",
    "CostFactors",
    "DosingUnit",
    "FerricDose",
    "FlowContext",
    "FlowParameterized",
    "FlowSetpoint",
    "HeatBalanceTemperature",
    "HydraulicDelayUnit",
    "IFASUnit",
    "IdealThickener",
    "IdentityTranslator",
    "InfluentFractions",
    "InfluentSeries",
    "InterfaceSettling",
    "LayeredSettling",
    "MBBRUnit",
    "MBRUnit",
    "MassBalance",
    "MixerUnit",
    "NoDigesterError",
    "OperatingCost",
    "PIController",
    "PTCResult",
    "ParameterLayout",
    "PiecewiseConstantSchedule",
    "Plant",
    "PlantCheck",
    "PlantObservable",
    "PlantSolution",
    "PrimaryClarifier",
    "RatioSplitter",
    "Reagent",
    "SBRPhase",
    "SBRUnit",
    "SetpointSplitter",
    "SettlingModel",
    "SludgeMetrics",
    "StateTranslator",
    "StatelessUnit",
    "SteadyStateResult",
    "StorageTank",
    "Stream",
    "StreamSeries",
    "TemperatureModel",
    "ThresholdSplitter",
    "UVUnit",
    "Unit",
    "UnknownPortError",
    "UnknownUnitError",
    "WiringError",
    "a2o_influent",
    "a2o_warm_start",
    "aeration_energy",
    "blower_airflow_total",
    "blower_energy",
    "blower_power_kw",
    "build_a2o",
    "calibrate_plant",
    "carbon_footprint",
    "carbon_mass",
    "characterize_influent",
    "co2e_from_energy",
    "ct_log_removal",
    "ct_value",
    "derived_BOD",
    "derived_COD",
    "derived_TKN",
    "derived_TSS",
    "design_summary",
    "direct_n2o_emission",
    "effluent_averages",
    "effluent_quality_index",
    "evaluate_bsm1",
    "evaluate_bsm2",
    "fractionate",
    "heating_energy",
    "mass_balance",
    "methane_to_co2e",
    "mixing_energy",
    "n2o_n_to_co2e",
    "operating_cost",
    "operational_cost_index",
    "operational_cost_index_bsm2",
    "oxygen_saturation",
    "ptc_forward",
    "pumping_energy",
    "pumping_energy_bsm2",
    "read_influent_csv",
    "required_airflow",
    "size_activated_sludge",
    "sludge_metrics",
    "solve_steady_state",
    "stripped_n2o",
    "t10_from_baffling",
    "t10_from_rtd",
    "uv_dose",
    "uv_log_inactivation",
]
