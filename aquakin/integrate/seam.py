"""The private ``aquakin.integrate`` internals that the plant layer builds on.

``Plant.calibrate``, ``plant_fisher_information`` and
``plant_profile_likelihood`` reuse the reactor calibration machinery instead
of copying it: the problem resolution helpers, the shared fit driver, the
objective and multistart, the Fisher-information helpers, the profile sweep and
the design criteria. The plant's solves likewise reuse the reactor solve
driver and ``atol`` coercion, the discrete adjoint's autonomization, the
biofilm's soluble mask, the DGSM aggregation and the species-by-name mixin.
Those pieces keep their leading underscore (they are not public API), and this
module names them in one place, so the contract between ``aquakin.integrate``
and ``aquakin.plant`` is a list someone can read rather than imports scattered
through the plant modules.

``aquakin.plant`` and ``aquakin.study`` import private names of
``aquakin.integrate`` only from here; ``tests/unit/test_integrate_seam.py``
checks that, and that every name listed here has a caller. Renaming or changing
one of these is a change to the plant layer too.
"""

from aquakin.integrate._solution import _HasNamedSpecies
from aquakin.integrate._solver import _coerce_atol, _run_diffeqsolve
from aquakin.integrate.biofilm import _default_soluble_mask
from aquakin.integrate.calibration.coords import _free_ic_fields, _resolve_ic_coords
from aquakin.integrate.calibration.driver import _check_start_args, _run_fit
from aquakin.integrate.calibration.loss import (
    _build_loss,
    _build_residual,
    _check_sigma_arg,
    _coerce_sigma,
    _convert_observations,
    _data_mean_square,
    _is_sigma_estimate,
    _n_obs_per_output,
    _observation_unit_factors,
    _observed_counts,
    _sigma_per_dataset,
)
from aquakin.integrate.calibration.multistart import _run_multistart
from aquakin.integrate.calibration.objective import _build_objective, _FitConfig
from aquakin.integrate.calibration.problem import (
    _array_digest,
    _CalibrationProblem,
    _check_start_in_domain,
    _is_multi_dataset,
    _resolve_priors,
    _resolve_rate_bounds,
)
from aquakin.integrate.discrete_adjoint import _autonomize
from aquakin.integrate.experiment_design import (
    _LARGER_IS_BETTER,
    _check_criterion,
    _criterion,
    _subset_indices,
)
from aquakin.integrate.fisher import _active_priors, _observation_noise, _resolve_transforms
from aquakin.integrate.global_sensitivity import _dgsm_aggregate
from aquakin.integrate.profile import (
    _check_profile_args,
    _cold_starts,
    _guarded,
    _point_optimizer,
    _profile_result,
    _profile_sweep,
)

__all__ = [
    "_LARGER_IS_BETTER",
    "_CalibrationProblem",
    "_FitConfig",
    "_HasNamedSpecies",
    "_active_priors",
    "_array_digest",
    "_autonomize",
    "_build_loss",
    "_build_objective",
    "_build_residual",
    "_check_criterion",
    "_check_profile_args",
    "_check_sigma_arg",
    "_check_start_args",
    "_check_start_in_domain",
    "_coerce_atol",
    "_coerce_sigma",
    "_cold_starts",
    "_convert_observations",
    "_criterion",
    "_data_mean_square",
    "_default_soluble_mask",
    "_dgsm_aggregate",
    "_free_ic_fields",
    "_guarded",
    "_is_multi_dataset",
    "_is_sigma_estimate",
    "_n_obs_per_output",
    "_observation_noise",
    "_observation_unit_factors",
    "_observed_counts",
    "_point_optimizer",
    "_profile_result",
    "_profile_sweep",
    "_resolve_ic_coords",
    "_resolve_priors",
    "_resolve_rate_bounds",
    "_resolve_transforms",
    "_run_diffeqsolve",
    "_run_fit",
    "_run_multistart",
    "_sigma_per_dataset",
    "_subset_indices",
]
