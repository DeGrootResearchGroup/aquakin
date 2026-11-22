"""Shared internals for the integrator submodules, re-exported in one place.

Not part of the public API. The code lives in four modules by concern:

- ``_config``: ``IntegratorConfig``, ``DifferentiationConfig``, the AD-mode
  helpers, the finite-gradient check and the ``Reactor`` protocols;
- ``_solution``: the solution mixins and the optional pandas / matplotlib
  imports;
- ``_cache``: the compiled-solver cache and its settings keys;
- ``_solver``: solve-time validation, ``atol`` and time-unit resolution, the
  friendly solve errors, and the implicit solver / controller builders.

This module re-exports every name, through ``__all__``, so existing imports
keep working; ``tests/unit/test_common_exports.py`` pins the list. These
modules import only from ``aquakin.core``, JAX and Diffrax, never from the rest
of ``aquakin.integrate``.
"""

from aquakin.core.tolerance import default_atol
from aquakin.integrate._cache import (
    _SOLVER_CACHE,
    _assemble_settings_key,
    atol_cache_key,
    cached_jitted_solver,
    concrete_settings_key,
    reactor_settings_key,
    settings_cache_key,
)
from aquakin.integrate._config import (
    ConditionedReactor,
    DifferentiationConfig,
    GradientCheckMixin,
    IntegratorConfig,
    Reactor,
    check_finite_gradient,
    forward_adjoint,
    with_adjoint,
)
from aquakin.integrate._solution import (
    ExportableSolutionMixin,
    PlottableSolutionMixin,
    _HasNamedSpecies,
    build_dataframe,
    require_matplotlib,
    require_pandas,
)
from aquakin.integrate._solver import (
    _CANONICAL_SOLVERS,
    _TIME_UNIT_SECONDS,
    NEWTON_TOL_FACTOR,
    _coerce_atol,
    _interp_fields_to_scalar,
    _run_diffeqsolve,
    build_implicit_solver,
    build_step_controller,
    friendly_solve_errors,
    init_solver_settings,
    is_forward_mode_ad_error,
    make_chemistry_rhs,
    native_time_factor,
    require_increasing_t_span,
    resolve_layered_atol,
    resolve_state_atol,
    solve_chemistry,
    to_native_time,
    validate_C0_params,
    validate_t_eval,
)

__all__ = [
    "NEWTON_TOL_FACTOR",
    "_CANONICAL_SOLVERS",
    "_SOLVER_CACHE",
    "_TIME_UNIT_SECONDS",
    "ConditionedReactor",
    "DifferentiationConfig",
    "ExportableSolutionMixin",
    "GradientCheckMixin",
    "IntegratorConfig",
    "PlottableSolutionMixin",
    "Reactor",
    "_HasNamedSpecies",
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
