"""Utility helpers: model rendering, RTD analytics, conservation and unit checks.

This subpackage is the **complete public utility surface**. It aggregates the
user-facing helpers from its submodules (``latex``, ``rtd``, ``balance``,
``composition``, ``units``) into one namespace, so ``from aquakin.utils import
check_conservation`` works alongside ``to_latex`` / ``E_curve``. The top-level
``aquakin`` namespace re-exports a *curated subset* of these (the ones in the
core model-analysis workflow -- ``check_conservation`` / ``check_nitrogen`` /
``parse_units`` / ``composition_table`` / ``canonical_content`` /
``check_model_units`` / ``UnitWarning``); the presentation/analytics helpers
(``to_latex``, the RTD curves) stay in this namespace, the same way the full set
of plant unit types stays in ``aquakin.plant``.

The lower-level building blocks (``conservation_residuals`` /
``nitrogen_residuals`` return raw residual arrays; ``check_rate_units`` operates
on a single rate AST; ``Dimension`` is the unit-algebra type) are intentionally
left importable from their submodules rather than aggregated here -- ``__all__``
is the curated utility API, not every public symbol.
"""

from aquakin.utils.balance import check_conservation, check_nitrogen
from aquakin.utils.composition import canonical_content, composition_table
from aquakin.utils.latex import to_latex
from aquakin.utils.rtd import (
    E_curve,
    F_curve,
    mean_residence_time,
    morrill_index,
    percentile_time,
    variance,
)
from aquakin.utils.units import UnitWarning, check_model_units, parse_units

__all__ = [
    "E_curve",
    "F_curve",
    "UnitWarning",
    "canonical_content",
    "check_conservation",
    "check_model_units",
    "check_nitrogen",
    "composition_table",
    "mean_residence_time",
    "morrill_index",
    "parse_units",
    "percentile_time",
    "to_latex",
    "variance",
]
