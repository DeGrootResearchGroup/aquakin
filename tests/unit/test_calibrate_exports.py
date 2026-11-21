"""``aquakin.integrate.calibrate`` keeps exporting every name it has exported.

The module was split into the ``aquakin.integrate.calibration`` subpackage, and
it re-exports what moved: tests, ``fisher.py``, ``profile.py`` and the plant
layer import private names from it, and a test monkeypatches
``_ReactorForwardModel`` on it. The list below is every aquakin-defined name the
module exported before the split (the stdlib / NumPy / JAX / SciPy helpers it
happened to import, such as ``np`` or ``least_squares``, are not part of it).
"""

import importlib

import pytest

EXPORTED = [
    "CalibrationResult",
    "DifferentiationConfig",
    "FreeICConfig",
    "LaplaceConfig",
    "OptimizerConfig",
    "PredictiveBand",
    "Reactor",
    "_AT_BOUND_ATOL",
    "_AT_BOUND_RTOL",
    "_CalibrationProblem",
    "_DEFAULT_IC_BOUNDS",
    "_FitConfig",
    "_IRLS_MAX_ITER",
    "_LaplaceOut",
    "_NULL_LOADING_TOL",
    "_NULL_RTOL",
    "_ObjectiveBundle",
    "_OptOut",
    "_ReactorForwardModel",
    "_SOLVE_FAILURE_HINT",
    "_SSE_FLOOR",
    "_VALID_LOSSES",
    "_VALID_OPTIMIZERS",
    "_array_digest",
    "_at_bound_named",
    "_at_bound_sides",
    "_bound_to_unconstrained",
    "_build_loss",
    "_build_masked_loss",
    "_build_masked_residual",
    "_build_objective",
    "_build_residual",
    "_check_sigma_arg",
    "_check_start_args",
    "_check_start_gradient",
    "_clip_start",
    "_coerce_sigma",
    "_convert_observations",
    "_data_term",
    "_fill_missing",
    "_fit_result",
    "_forward_jac",
    "_free_ic_fields",
    "_from_unconstrained",
    "_ic_coordinate_names",
    "_ic_theta_from_pools",
    "_initial_point",
    "_is_sigma_estimate",
    "_is_solve_failure",
    "_jacobian_physical_wrt_theta",
    "_joint_names",
    "_laplace_covariance",
    "_laplace_posterior",
    "_masked_parts",
    "_missing_mask",
    "_n_obs_per_output",
    "_noise_estimate",
    "_observation_unit_factors",
    "_observed_counts",
    "_optimizer_bounds",
    "_per_species_ic_bounds",
    "_pooled_sse",
    "_predict",
    "_residual_parts",
    "_resolve_ic_coords",
    "_resolve_laplace",
    "_resolve_problem",
    "_resolve_rate_bounds",
    "_resolve_starts",
    "_run_fit",
    "_run_multistart",
    "_sigma_per_dataset",
    "_simplex_fraction_cov",
    "_simplex_pools",
    "_to_unconstrained",
    "calibrate",
    "check_finite_gradient",
    "dphysical_dunconstrained",
    "forward_adjoint",
    "from_unconstrained",
    "native_time_factor",
    "observation_size",
    "observation_takes_time",
    "observe",
    "to_unconstrained",
    "with_adjoint",
]


def _calibrate_module():
    # ``aquakin.integrate.calibrate`` as an attribute is the function, so go
    # through the import system for the module.
    return importlib.import_module("aquakin.integrate.calibrate")


@pytest.mark.parametrize("name", EXPORTED)
def test_name_still_importable(name):
    """Each name is an attribute of the module, as ``from ... import`` needs."""
    assert hasattr(_calibrate_module(), name), f"aquakin.integrate.calibrate lost {name!r}"


def test_all_lists_exactly_the_exported_names():
    """``__all__`` is the re-export list; it names exactly the pinned set, so a
    name dropped from it (and with it the import that kept it alive) fails."""
    assert sorted(_calibrate_module().__all__) == sorted(EXPORTED)
