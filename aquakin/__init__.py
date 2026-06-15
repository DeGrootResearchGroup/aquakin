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
    _overrides_explicit_choice = (
        _jax_already_imported or _env_x64 in ("0", "false", "off", "no")
    )
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

from aquakin.core.conditions import OperatingConditions, SpatialConditions
from aquakin.core.network import CompiledNetwork, compile_network
from aquakin.core.parser import parse_rate_expression
from aquakin.integrate.batch import BatchReactor, BatchSolution
from aquakin.integrate.biofilm import BiofilmReactor, BiofilmSolution
from aquakin.integrate.calibrate import CalibrationResult, PredictiveBand, calibrate
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
from aquakin.integrate._common import check_finite_gradient, forward_adjoint
from aquakin.integrate.forward_sensitivity import (
    ForwardSensitivityResult,
    forward_sensitivity,
)
from aquakin.integrate.pfr import PFRSolution, PlugFlowReactor
from aquakin.integrate.sensitivity import (
    DGSMResult,
    FitResult,
    SensitivityResult,
    dgsm,
    fit,
    sensitivity,
)
from aquakin.integrate.experiments import (
    Constraint,
    KPIComparison,
    MonteCarloResult,
    OptimizeResult,
    ScenarioComparison,
    compare_scenarios,
    kpi_comparison,
    monte_carlo,
    optimize_design,
)
from aquakin.plant import (
    ActivatedSludgeSizing,
    Aeration,
    BSM1Evaluation,
    BSM2Evaluation,
    CSTRUnit,
    CarbonFootprint,
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
    SludgeMetrics,
    SplitterUnit,
    StateTranslator,
    Stream,
    Unit,
    aeration_energy,
    carbon_footprint,
    carbon_mass,
    co2e_from_energy,
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
    size_activated_sludge,
    sludge_metrics,
    stripped_n2o,
)
from aquakin.schema.loader import load_network, load_network_from_file
from aquakin.utils.balance import check_conservation
from aquakin.utils.composition import (
    canonical_content,
    composition_table,
)
from aquakin.utils.units import (
    UnitWarning,
    check_network_units,
    parse_units,
)

__all__ = [
    "ActivatedSludgeSizing",
    "BSM1Evaluation",
    "BSM2Evaluation",
    "BatchReactor",
    "BatchSolution",
    "BiofilmReactor",
    "BiofilmSolution",
    "CFDReactor",
    "Aeration",
    "CSTRUnit",
    "CalibrationResult",
    "PredictiveBand",
    "ProfileResult",
    "CompiledNetwork",
    "FitResult",
    "IdentityTranslator",
    "InfluentFractions",
    "InfluentSeries",
    "characterize_influent",
    "fractionate",
    "read_influent_csv",
    "MixerUnit",
    "PFRSolution",
    "ParticleTrackReactor",
    "Plant",
    "PlantCheck",
    "PlantSolution",
    "PlugFlowReactor",
    "ActivatedSludgeSizing",
    "SludgeMetrics",
    "SplitterUnit",
    "StateTranslator",
    "Stream",
    "Unit",
    "DGSMResult",
    "ForwardSensitivityResult",
    "SensitivityResult",
    "OperatingConditions",
    "SpatialConditions",
    "Track",
    "TrackSolution",
    "aeration_energy",
    "calibrate",
    "carbon_mass",
    "canonical_content",
    "check_conservation",
    "composition_table",
    "check_network_units",
    "parse_units",
    "UnitWarning",
    "profile_likelihood",
    "compile_network",
    "derived_BOD",
    "derived_COD",
    "derived_TKN",
    "derived_TSS",
    "dgsm",
    "monte_carlo",
    "compare_scenarios",
    "kpi_comparison",
    "optimize_design",
    "MonteCarloResult",
    "ScenarioComparison",
    "KPIComparison",
    "OptimizeResult",
    "Constraint",
    "CarbonFootprint",
    "CostFactors",
    "OperatingCost",
    "carbon_footprint",
    "co2e_from_energy",
    "direct_n2o_emission",
    "methane_to_co2e",
    "n2o_n_to_co2e",
    "operating_cost",
    "stripped_n2o",
    "effluent_averages",
    "effluent_quality_index",
    "check_finite_gradient",
    "esdirk_adjoint_solve",
    "evaluate_bsm1",
    "evaluate_bsm2",
    "fit",
    "forward_adjoint",
    "forward_sensitivity",
    "heating_energy",
    "implicit_euler_adjoint_solve",
    "integrate_ensemble",
    "load_network",
    "load_network_from_file",
    "ComponentBalance",
    "MassBalance",
    "mass_balance",
    "mixing_energy",
    "operational_cost_index",
    "operational_cost_index_bsm2",
    "parse_rate_expression",
    "pumping_energy",
    "pumping_energy_bsm2",
    "sensitivity",
    "size_activated_sludge",
    "sludge_metrics",
]

__version__ = "0.1.0"
