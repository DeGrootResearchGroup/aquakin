"""``aquakin.integrate._common`` keeps exporting every name it has exported.

The module was split by concern into ``integrate/_config.py``,
``_solution.py``, ``_cache.py`` and ``_solver.py``, and it re-exports what
moved: about forty modules across ``integrate``, ``plant`` and the tests import
from it, and ``tests/integration/test_compile_cache.py`` reads its
``_SOLVER_CACHE``. The list below is every name defined in the module before
the split, plus ``default_atol``, which callers import through it.
"""

import importlib

import pytest

EXPORTED = [
    "ConditionedReactor",
    "DifferentiationConfig",
    "ExportableSolutionMixin",
    "GradientCheckMixin",
    "IntegratorConfig",
    "NEWTON_TOL_FACTOR",
    "PlottableSolutionMixin",
    "Reactor",
    "_CANONICAL_SOLVERS",
    "_HasNamedSpecies",
    "_SOLVER_CACHE",
    "_TIME_UNIT_SECONDS",
    "_assemble_settings_key",
    "_coerce_atol",
    "_interp_fields_to_scalar",
    "_run_diffeqsolve",
    "atol_cache_key",
    "build_dataframe",
    "build_implicit_solver",
    "build_step_controller",
    "cached_jitted_solver",
    "check_finite_gradient",
    "concrete_settings_key",
    "default_atol",
    "forward_adjoint",
    "friendly_solve_errors",
    "init_solver_settings",
    "is_forward_mode_ad_error",
    "make_chemistry_rhs",
    "native_time_factor",
    "reactor_settings_key",
    "require_increasing_t_span",
    "require_matplotlib",
    "require_pandas",
    "resolve_layered_atol",
    "resolve_state_atol",
    "settings_cache_key",
    "solve_chemistry",
    "to_native_time",
    "validate_C0_params",
    "validate_t_eval",
    "with_adjoint",
]


def _common_module():
    return importlib.import_module("aquakin.integrate._common")


@pytest.mark.parametrize("name", EXPORTED)
def test_name_still_importable(name):
    """Each name is an attribute of the module, as ``from ... import`` needs."""
    assert hasattr(_common_module(), name), f"aquakin.integrate._common lost {name!r}"


def test_all_lists_exactly_the_exported_names():
    """``__all__`` is the re-export list; it names exactly the pinned set, so a
    name dropped from it (and with it the import that kept it alive) fails."""
    assert sorted(_common_module().__all__) == sorted(EXPORTED)
