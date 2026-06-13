"""aquakin: reactive scalar transport kinetics for aqueous environmental systems."""

import jax

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
from aquakin.integrate._common import forward_adjoint
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
from aquakin.plant import (
    ActivatedSludgeSizing,
    Aeration,
    BSM1Evaluation,
    BSM2Evaluation,
    CSTRUnit,
    IdentityTranslator,
    InfluentSeries,
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
    carbon_mass,
    derived_BOD,
    derived_COD,
    derived_TKN,
    derived_TSS,
    effluent_averages,
    effluent_quality_index,
    evaluate_bsm1,
    evaluate_bsm2,
    heating_energy,
    mixing_energy,
    operational_cost_index,
    operational_cost_index_bsm2,
    pumping_energy,
    pumping_energy_bsm2,
    size_activated_sludge,
    sludge_metrics,
)
from aquakin.schema.loader import load_network, load_network_from_file
from aquakin.utils.balance import check_conservation
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
    "InfluentSeries",
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
    "check_conservation",
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
    "effluent_averages",
    "effluent_quality_index",
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
