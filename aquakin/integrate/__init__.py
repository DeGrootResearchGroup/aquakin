"""ODE integration over compiled reaction models (Diffrax-backed).

This subpackage is the **complete public integration surface**: reactors,
solutions, calibration/fitting, sensitivity (forward, adjoint, DGSM, profile),
events, design optimisation, Monte-Carlo and scenario comparison. Every public
name here is also re-exported by the top-level ``aquakin`` namespace, which
carries a curated subset for convenience -- import the full set from
``aquakin.integrate`` (mirroring ``aquakin.plant``), or the common entry points
from ``aquakin`` directly.
"""

from aquakin.integrate._common import (
    DifferentiationConfig,
    IntegratorConfig,
    check_finite_gradient,
    forward_adjoint,
)
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
from aquakin.integrate.cfd import CFDReactor
from aquakin.integrate.design import Constraint, OptimizeResult, optimize_design
from aquakin.integrate.discrete_adjoint import (
    esdirk_adjoint_solve,
    implicit_euler_adjoint_solve,
)
from aquakin.integrate.events import Event, EventedResult, solve_with_events
from aquakin.integrate.fit import FitResult, fit
from aquakin.integrate.forward_sensitivity import (
    ForwardSensitivityResult,
    forward_sensitivity,
)
from aquakin.integrate.global_sensitivity import DGSMResult, dgsm
from aquakin.integrate.monte_carlo import MonteCarloResult, monte_carlo
from aquakin.integrate.particle import (
    ParticleTrackReactor,
    Track,
    TrackSolution,
    integrate_ensemble,
)
from aquakin.integrate.pfr import PFRSolution, PlugFlowReactor
from aquakin.integrate.profile import ProfileResult, profile_likelihood
from aquakin.integrate.scenarios import (
    KPIComparison,
    ScenarioComparison,
    compare_scenarios,
    kpi_comparison,
)
from aquakin.integrate.sensitivity import SensitivityResult, sensitivity

__all__ = [
    "BatchReactor",
    "BatchSolution",
    "BiofilmReactor",
    "BiofilmSolution",
    "CFDReactor",
    "CalibrationResult",
    "Constraint",
    "DGSMResult",
    "DifferentiationConfig",
    "Event",
    "EventedResult",
    "FitResult",
    "ForwardSensitivityResult",
    "FreeICConfig",
    "IntegratorConfig",
    "KPIComparison",
    "LaplaceConfig",
    "MonteCarloResult",
    "OptimizeResult",
    "OptimizerConfig",
    "PFRSolution",
    "ParticleTrackReactor",
    "PlugFlowReactor",
    "PredictiveBand",
    "ProfileResult",
    "ScenarioComparison",
    "SensitivityResult",
    "Track",
    "TrackSolution",
    "calibrate",
    "check_finite_gradient",
    "compare_scenarios",
    "dgsm",
    "esdirk_adjoint_solve",
    "fit",
    "forward_adjoint",
    "forward_sensitivity",
    "implicit_euler_adjoint_solve",
    "integrate_ensemble",
    "kpi_comparison",
    "monte_carlo",
    "optimize_design",
    "profile_likelihood",
    "sensitivity",
    "solve_with_events",
]
