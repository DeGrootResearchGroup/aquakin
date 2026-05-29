"""ODE integration over compiled reaction networks (Diffrax-backed)."""

from aquakin.integrate.batch import BatchReactor, BatchSolution
from aquakin.integrate.calibrate import CalibrationResult, calibrate
from aquakin.integrate.cfd import CFDReactor
from aquakin.integrate.particle import (
    ParticleTrackReactor,
    Track,
    TrackSolution,
    integrate_ensemble,
)
from aquakin.integrate.pfr import PFRSolution, PlugFlowReactor
from aquakin.integrate.sensitivity import SensitivityResult, fit, sensitivity

__all__ = [
    "BatchReactor",
    "BatchSolution",
    "CFDReactor",
    "CalibrationResult",
    "PFRSolution",
    "ParticleTrackReactor",
    "PlugFlowReactor",
    "SensitivityResult",
    "Track",
    "TrackSolution",
    "calibrate",
    "fit",
    "integrate_ensemble",
    "sensitivity",
]
