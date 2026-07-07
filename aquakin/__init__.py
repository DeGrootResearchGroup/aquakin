"""aquakin: reactive scalar transport kinetics for aqueous environmental systems."""

import os as _os
import sys as _sys
import warnings as _warnings

# aquakin enables JAX 64-bit (x64) mode at import: the stiff implicit ODE solves
# need double precision (repeatedly forming small net differences in a stiff
# Newton step loses the accurate digits at float32). This mutates GLOBAL,
# process-wide JAX state -- any other JAX code in the same process will use
# float64 afterward. The behaviour is deliberate and documented, but to keep it
# from being a *silent* surprise we warn when we are overriding what looks like
# an explicit float32 preference: JAX was already imported (so other code may be
# relying on the default float32), or JAX_ENABLE_X64 is set to a false value.
_jax_already_imported = "jax" in _sys.modules

import jax

if not jax.config.jax_enable_x64:
    _env_x64 = _os.environ.get("JAX_ENABLE_X64", "").strip().lower()
    _overrides_explicit_choice = _jax_already_imported or _env_x64 in ("0", "false", "off", "no")
    if _overrides_explicit_choice:
        _warnings.warn(
            "aquakin is enabling JAX 64-bit (x64) mode, which it requires for "
            "stiff ODE integration. This is GLOBAL, process-wide JAX state: "
            "other JAX code in this process will now use float64. It overrides "
            "an apparent float32 preference (JAX was already imported, or "
            "JAX_ENABLE_X64 is set off). Enable x64 yourself before importing "
            "aquakin, or run aquakin in a separate process, to silence this.",
            stacklevel=2,
        )
    jax.config.update("jax_enable_x64", True)

import logging as _logging

# Library logging hygiene: attach a NullHandler to the package's root logger so
# aquakin never configures logging on the application's behalf (it emits nothing
# unless the caller opts in) and there is no "No handlers could be found" noise.
# Progress output -- e.g. the ``progress=`` sampling loops in
# ``plant.steady_state_dgsm`` / ``dynamic_dgsm`` -- is logged at INFO on child
# loggers; a caller sees it by enabling INFO, e.g.
# ``logging.basicConfig(level=logging.INFO)`` or
# ``logging.getLogger("aquakin").setLevel(logging.INFO)`` with a handler.
_logging.getLogger("aquakin").addHandler(_logging.NullHandler())

from aquakin.core.conditions import OperatingConditions, SpatialConditions
from aquakin.core.model import CompiledModel, compile_model
from aquakin.core.parser import parse_rate_expression
from aquakin.integrate.batch import BatchReactor, BatchSolution
from aquakin.integrate.biofilm import BiofilmReactor, BiofilmSolution
from aquakin.integrate.calibrate import (
    CalibrationResult,
    FreeICConfig,
    LaplaceConfig,
    OptimizerConfig,
    PredictiveBand,
    calibrate,
)
from aquakin.integrate.profile import ProfileResult, profile_likelihood
from aquakin.integrate.cfd import CFDReactor
from aquakin.integrate.particle import (
    ParticleTrackReactor,
    Track,
    TrackSolution,
    integrate_ensemble,
)
from aquakin.integrate.discrete_adjoint import (
    esdirk_adjoint_solve,
    implicit_euler_adjoint_solve,
)
from aquakin.integrate._common import (
    DifferentiationConfig,
    IntegratorConfig,
    check_finite_gradient,
    forward_adjoint,
)
from aquakin.integrate.forward_sensitivity import (
    ForwardSensitivityResult,
    forward_sensitivity,
)
from aquakin.integrate.pfr import PFRSolution, PlugFlowReactor
from aquakin.integrate.sensitivity import SensitivityResult, sensitivity
from aquakin.integrate.fit import FitResult, fit
from aquakin.integrate.global_sensitivity import DGSMResult, dgsm
from aquakin.integrate.events import Event, EventedResult, solve_with_events
from aquakin.integrate.design import Constraint, OptimizeResult, optimize_design
from aquakin.integrate.monte_carlo import MonteCarloResult, monte_carlo
from aquakin.integrate.scenarios import (
    KPIComparison,
    ScenarioComparison,
    compare_scenarios,
    kpi_comparison,
)
from aquakin.plant import (
    ActivatedSludgeSizing,
    Aeration,
    AerationDesignPoint,
    AerationSystem,
    AlgebraicTemperature,
    BSM1Evaluation,
    BSM2Evaluation,
    CSTRUnit,
    CarbonFootprint,
    ChlorineContactUnit,
    ComponentBalance,
    CostFactors,
    IdentityTranslator,
    InfluentFractions,
    InfluentSeries,
    MassBalance,
    OperatingCost,
    characterize_influent,
    fractionate,
    mass_balance,
    read_influent_csv,
    MixerUnit,
    Plant,
    PlantCheck,
    PlantSolution,
    HeatBalanceTemperature,
    RatioSplitter,
    SetpointSplitter,
    SludgeMetrics,
    StateTranslator,
    Stream,
    TemperatureModel,
    ThresholdSplitter,
    UVUnit,
    Unit,
    aeration_energy,
    blower_airflow_total,
    blower_energy,
    blower_power_kw,
    carbon_footprint,
    carbon_mass,
    co2e_from_energy,
    ct_log_removal,
    ct_value,
    design_summary,
    derived_BOD,
    derived_COD,
    derived_TKN,
    derived_TSS,
    direct_n2o_emission,
    effluent_averages,
    effluent_quality_index,
    evaluate_bsm1,
    evaluate_bsm2,
    heating_energy,
    methane_to_co2e,
    mixing_energy,
    n2o_n_to_co2e,
    operating_cost,
    operational_cost_index,
    operational_cost_index_bsm2,
    pumping_energy,
    pumping_energy_bsm2,
    required_airflow,
    size_activated_sludge,
    sludge_metrics,
    stripped_n2o,
    t10_from_baffling,
    t10_from_rtd,
    uv_dose,
    uv_log_inactivation,
)
from aquakin.schema.loader import (
    clear_model_cache,
    load_model,
    load_model_from_file,
)
from aquakin.utils.balance import check_conservation
from aquakin.utils.composition import (
    canonical_content,
    composition_table,
)
from aquakin.utils.units import (
    UnitWarning,
    check_model_units,
    parse_units,
)

__all__ = [
    "ActivatedSludgeSizing",
    "ActivatedSludgeSizing",
    "Aeration",
    "AerationDesignPoint",
    "AerationSystem",
    "AlgebraicTemperature",
    "BSM1Evaluation",
    "BSM2Evaluation",
    "BatchReactor",
    "BatchSolution",
    "BiofilmReactor",
    "BiofilmSolution",
    "CFDReactor",
    "CSTRUnit",
    "CalibrationResult",
    "CarbonFootprint",
    "ChlorineContactUnit",
    "CompiledModel",
    "ComponentBalance",
    "Constraint",
    "CostFactors",
    "DGSMResult",
    "DifferentiationConfig",
    "Event",
    "EventedResult",
    "FitResult",
    "ForwardSensitivityResult",
    "FreeICConfig",
    "HeatBalanceTemperature",
    "IdentityTranslator",
    "InfluentFractions",
    "InfluentSeries",
    "IntegratorConfig",
    "KPIComparison",
    "LaplaceConfig",
    "MassBalance",
    "MixerUnit",
    "MonteCarloResult",
    "OperatingConditions",
    "OperatingCost",
    "OptimizeResult",
    "OptimizerConfig",
    "PFRSolution",
    "ParticleTrackReactor",
    "Plant",
    "PlantCheck",
    "PlantSolution",
    "PlugFlowReactor",
    "PredictiveBand",
    "ProfileResult",
    "RatioSplitter",
    "ScenarioComparison",
    "SensitivityResult",
    "SetpointSplitter",
    "SludgeMetrics",
    "SpatialConditions",
    "StateTranslator",
    "Stream",
    "TemperatureModel",
    "ThresholdSplitter",
    "Track",
    "TrackSolution",
    "UVUnit",
    "Unit",
    "UnitWarning",
    "aeration_energy",
    "blower_airflow_total",
    "blower_energy",
    "blower_power_kw",
    "calibrate",
    "canonical_content",
    "carbon_footprint",
    "carbon_mass",
    "characterize_influent",
    "check_conservation",
    "check_finite_gradient",
    "check_model_units",
    "clear_model_cache",
    "co2e_from_energy",
    "compare_scenarios",
    "compile_model",
    "composition_table",
    "ct_log_removal",
    "ct_value",
    "derived_BOD",
    "derived_COD",
    "derived_TKN",
    "derived_TSS",
    "design_summary",
    "dgsm",
    "direct_n2o_emission",
    "effluent_averages",
    "effluent_quality_index",
    "esdirk_adjoint_solve",
    "evaluate_bsm1",
    "evaluate_bsm2",
    "fit",
    "forward_adjoint",
    "forward_sensitivity",
    "fractionate",
    "heating_energy",
    "implicit_euler_adjoint_solve",
    "integrate_ensemble",
    "kpi_comparison",
    "load_model",
    "load_model_from_file",
    "mass_balance",
    "methane_to_co2e",
    "mixing_energy",
    "monte_carlo",
    "n2o_n_to_co2e",
    "operating_cost",
    "operational_cost_index",
    "operational_cost_index_bsm2",
    "optimize_design",
    "parse_rate_expression",
    "parse_units",
    "profile_likelihood",
    "pumping_energy",
    "pumping_energy_bsm2",
    "read_influent_csv",
    "required_airflow",
    "sensitivity",
    "size_activated_sludge",
    "sludge_metrics",
    "solve_with_events",
    "stripped_n2o",
    "t10_from_baffling",
    "t10_from_rtd",
    "uv_dose",
    "uv_log_inactivation",
]

__version__ = "0.1.0"
