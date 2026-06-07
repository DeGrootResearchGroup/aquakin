"""aquakin: reactive scalar transport kinetics for aqueous environmental systems."""

import jax

jax.config.update("jax_enable_x64", True)

from aquakin.core.conditions import SpatialConditions
from aquakin.core.network import CompiledNetwork, compile_network
from aquakin.core.parser import parse_rate_expression
from aquakin.integrate.batch import BatchReactor, BatchSolution
from aquakin.integrate.calibrate import CalibrationResult, PredictiveBand, calibrate
from aquakin.integrate.profile import ProfileResult, profile_likelihood
from aquakin.integrate.cfd import CFDReactor
from aquakin.integrate.particle import (
    ParticleTrackReactor,
    Track,
    TrackSolution,
    integrate_ensemble,
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
    CSTRUnit,
    IdentityTranslator,
    InfluentSeries,
    MixerUnit,
    Plant,
    PlantSolution,
    SplitterUnit,
    StateTranslator,
    Stream,
    Unit,
)
from aquakin.schema.loader import load_network, load_network_from_file

__all__ = [
    "BatchReactor",
    "BatchSolution",
    "CFDReactor",
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
    "PlantSolution",
    "PlugFlowReactor",
    "SplitterUnit",
    "StateTranslator",
    "Stream",
    "Unit",
    "DGSMResult",
    "SensitivityResult",
    "SpatialConditions",
    "Track",
    "TrackSolution",
    "calibrate",
    "profile_likelihood",
    "compile_network",
    "dgsm",
    "fit",
    "integrate_ensemble",
    "load_network",
    "load_network_from_file",
    "parse_rate_expression",
    "sensitivity",
]

__version__ = "0.1.0"
