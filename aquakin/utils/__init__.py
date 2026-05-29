"""Utility helpers (LaTeX rendering, RTD analytics, etc.)."""

from aquakin.utils.latex import to_latex
from aquakin.utils.rtd import (
    E_curve,
    F_curve,
    mean_residence_time,
    morrill_index,
    percentile_time,
    variance,
)

__all__ = [
    "E_curve",
    "F_curve",
    "mean_residence_time",
    "morrill_index",
    "percentile_time",
    "to_latex",
    "variance",
]
