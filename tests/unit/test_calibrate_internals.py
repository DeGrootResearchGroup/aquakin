"""Unit tests for :func:`calibrate`'s extracted building blocks.

The god-function refactor split ``calibrate`` into independently testable pieces:
:func:`_resolve_problem` (validation + coercion), :class:`_CalibrationProblem`
(the resolved static problem), :func:`_optimizer_bounds`, and :func:`_forward_jac`
(the AD-direction rule). These exercise them directly -- no fit, no solve -- which
the previous single-closure structure could not. Behaviour equivalence to the old
monolith is covered by the full-fit tests in
``tests/integration/test_calibrate.py``; here we pin the units.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate._common import DifferentiationConfig
from aquakin.integrate.calibrate import (
    CalibrationResult,
    _build_loss,
    _build_residual,
    _CalibrationProblem,
    _FitConfig,
    _forward_jac,
    _free_ic_fields,
    _laplace_covariance,
    _optimizer_bounds,
    _resolve_laplace,
    _resolve_problem,
    calibrate,
)


def _resolve(model, C0, obs, t_obs, free, **overrides):
    """Call _resolve_problem with the calibrate defaults, overriding as needed."""
    kw = dict(
        transforms=None,
        initial_params=None,
        observed_species=None,
        time_unit=None,
        loss="mse",
        sigma=None,
        priors=None,
        use_priors=False,
        free_ic=None,
        ic_bounds=(1e-3, 1e4),
        ic_prior_log_std=None,
        param_halfwidth=None,
    )
    kw.update(overrides)
    return _resolve_problem(model, C0, obs, t_obs, free, **kw)


@pytest.fixture
def base(simple_model):
    """A minimal single-dataset problem for the A -> B decay model."""
    m = simple_model
    C0 = jnp.array([1.0, 0.0])
    t_obs = jnp.array([0.0, 1.0, 2.0])
    obs = jnp.zeros((3, m.n_species))
    return m, C0, obs, t_obs, [m.parameters[0]]


# ---------- _resolve_problem: coercion ----------


def test_resolve_problem_coerces_single_dataset(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free)
    assert isinstance(prob, _CalibrationProblem)
    assert prob.n_rate == 1
    assert prob.n_datasets == 1
    assert prob.free_params == free
    assert prob.n_observed == m.n_species
    assert len(prob.dataset_static) == 1
    assert not prob.has_priors
    assert prob.m_ic == 0
    # free_indices maps the name to its flat parameter index.
    assert int(prob.free_indices[0]) == m.param_index[free[0]]


def test_resolve_problem_transform_falls_back_to_model_declaration(base):
    m, C0, obs, t_obs, free = base
    # No explicit transform -> the parameter's declared transform (or "none").
    prob = _resolve(m, C0, obs, t_obs, free)
    assert prob.transforms[0] == m.parameter_transforms.get(free[0], "none")
    # An explicit override wins.
    prob2 = _resolve(m, C0, obs, t_obs, free, transforms={free[0]: "positive_log"})
    assert prob2.transforms[0] == "positive_log"


def test_resolve_problem_multi_dataset(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, [C0, C0], [obs, obs], [t_obs, t_obs], free)
    assert prob.n_datasets == 2
    assert len(prob.dataset_static) == 2
    assert len(prob.C0_base) == 2


def test_resolve_problem_priors_align_to_free_params(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free, priors={free[0]: (2.0, 0.5)})
    assert prob.has_priors
    assert prob.active_priors[free[0]] == (2.0, 0.5)
    assert float(prob.prior_mean[0]) == 2.0
    assert float(prob.prior_std[0]) == 0.5
    assert float(prob.prior_mask[0]) == 1.0


def test_resolve_problem_free_ic_centers_at_log_initial(base):
    m, C0, obs, t_obs, free = base
    sp = m.species[0]
    prob = _resolve(m, C0, obs, t_obs, free, free_ic=[sp])
    assert prob.m_ic == 1
    assert prob.free_ic == [sp]
    # The ic-prior centre is log of the (clipped) initial pool.
    expected = np.log(np.clip(float(C0[m.species_index[sp]]), 1e-3, 1e4))
    assert float(prob.ic_center_full[0]) == pytest.approx(expected)


# ---------- _resolve_problem: validation ----------


def test_resolve_problem_rejects_unknown_param(base):
    m, C0, obs, t_obs, _free = base
    with pytest.raises(KeyError, match="Unknown parameter"):
        _resolve(m, C0, obs, t_obs, ["not_a_param"])


def test_resolve_problem_rejects_bad_positive_log_initial(base):
    m, C0, obs, t_obs, free = base
    bad = m.default_parameters().at[m.param_index[free[0]]].set(-1.0)
    with pytest.raises(ValueError, match="positive_log"):
        _resolve(m, C0, obs, t_obs, free, transforms={free[0]: "positive_log"}, initial_params=bad)


def test_resolve_problem_rejects_nonascending_tobs(base):
    m, C0, obs, _t, free = base
    with pytest.raises(ValueError, match="strictly ascending"):
        _resolve(m, C0, obs, jnp.array([0.0, 2.0, 1.0]), free)


def test_resolve_problem_rejects_mismatched_dataset_lengths(base):
    m, C0, obs, t_obs, free = base
    with pytest.raises(ValueError, match="equal length"):
        _resolve(m, [C0, C0], [obs], [t_obs, t_obs], free)


def test_resolve_problem_rejects_unknown_free_ic_species(base):
    m, C0, obs, t_obs, free = base
    with pytest.raises(KeyError, match="Unknown free_ic species"):
        _resolve(m, C0, obs, t_obs, free, free_ic=["not_a_species"])


# ---------- _CalibrationProblem methods ----------


def test_problem_physical_from_theta_inverts_rate_theta0(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free, transforms={free[0]: "positive_log"})
    # rate_theta0 is the unconstrained start; physical_from_theta maps it back.
    physical = prob.physical_from_theta(prob.rate_theta0())
    assert float(physical[0]) == pytest.approx(float(prob.p0_full[prob.free_indices[0]]))


def test_problem_struct_key_is_shape_sensitive(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free)
    k = prob.struct_key("jax_adjoint")
    # The gradient backend is part of the key (so the two backends never collide).
    assert prob.struct_key("stable_adjoint") != k
    # A different observation-count problem keys differently.
    prob2 = _resolve(m, C0, obs[:, :1], t_obs, free, observed_species=[m.species[0]])
    assert prob2.struct_key("jax_adjoint") != k


# ---------- _optimizer_bounds ----------


def test_optimizer_bounds_unbounded_by_default(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free)
    bounds, lb, ub, ls_bounds, has = _optimizer_bounds(prob, prob.rate_theta0())
    assert bounds is None and not has
    assert not np.isfinite(lb[0]) and not np.isfinite(ub[0])


def test_optimizer_bounds_param_halfwidth_boxes_the_rates(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free, param_halfwidth=1.5)
    theta0 = np.asarray(prob.rate_theta0(), dtype=float)
    bounds, lb, ub, _ls, has = _optimizer_bounds(prob, prob.rate_theta0())
    assert has and bounds is not None
    assert lb[0] == pytest.approx(theta0[0] - 1.5)
    assert ub[0] == pytest.approx(theta0[0] + 1.5)


def test_optimizer_bounds_free_ic_log_bounded(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free, free_ic=[m.species[0]], ic_bounds=(1e-2, 10.0))
    bounds, lb, ub, _ls, has = _optimizer_bounds(prob, prob.rate_theta0())
    assert has
    # One rate dim (unbounded) + one free-IC dim (log-bounded by ic_bounds).
    assert lb[-1] == pytest.approx(np.log(1e-2))
    assert ub[-1] == pytest.approx(np.log(10.0))


# ---------- _forward_jac ----------


class _StubForwardModel:
    def __init__(self, forward_capable):
        self._cap = forward_capable

    def forward_capable(self):
        return self._cap


def _cfg(ad_mode, gradient):
    return _FitConfig(
        gradient=gradient,
        ad_mode=ad_mode,
        check_finite=True,
        stable_adjoint_max_steps=1,
        stable_adjoint_low_memory=False,
        optimizer="gauss_newton",
        max_iter=1,
        tol=1e-6,
        n_starts=1,
        jitter=0.5,
        jitter_schedule=None,
        seed=0,
        laplace=False,
        laplace_method="fd",
        laplace_ridge=1e-6,
        laplace_eig_keep=1e-2,
        laplace_fd_step=1e-3,
        laplace_dtmax=None,
        compiled_cache=None,
    )


@pytest.mark.parametrize(
    "ad_mode,gradient,cap,expected",
    [
        ("forward", "jax_adjoint", False, True),  # forced forward
        ("reverse", "jax_adjoint", True, False),  # forced reverse
        ("auto", "jax_adjoint", True, True),  # auto + forward-capable reactor
        ("auto", "jax_adjoint", False, False),  # auto + reverse-only reactor
        ("auto", "stable_adjoint", True, False),  # stable adjoint is reverse-only
    ],
)
def test_forward_jac_rule(ad_mode, gradient, cap, expected):
    assert _forward_jac(_cfg(ad_mode, gradient), _StubForwardModel(cap)) is expected


# ---------- _build_loss / _build_residual: argument validation ----------


def test_build_loss_wmse_requires_sigma():
    obs = jnp.zeros((3, 1))
    with pytest.raises(ValueError, match="loss='wmse' requires a sigma"):
        _build_loss("wmse", obs, None)


def test_build_loss_nll_requires_sigma():
    obs = jnp.zeros((3, 1))
    with pytest.raises(ValueError, match="loss='nll' requires a sigma"):
        _build_loss("nll", obs, None)


def test_build_loss_rejects_unknown_loss():
    obs = jnp.zeros((3, 1))
    with pytest.raises(ValueError, match="Unknown loss 'bogus'"):
        _build_loss("bogus", obs, None)


def test_build_residual_wmse_requires_sigma():
    obs = jnp.zeros((3, 1))
    with pytest.raises(ValueError, match="loss='wmse' requires a sigma"):
        _build_residual("wmse", obs, None)


def test_build_residual_nll_requires_sigma():
    obs = jnp.zeros((3, 1))
    with pytest.raises(ValueError, match="loss='nll' requires a sigma"):
        _build_residual("nll", obs, None)


def test_build_residual_rejects_unknown_loss():
    obs = jnp.zeros((3, 1))
    with pytest.raises(ValueError, match="Unknown loss 'bogus'"):
        _build_residual("bogus", obs, None)


# ---------- config normalisers: _resolve_laplace / _free_ic_fields ----------


def test_resolve_laplace_rejects_bad_type():
    with pytest.raises(TypeError, match="laplace must be a bool or LaplaceConfig"):
        _resolve_laplace("yes")


def test_free_ic_fields_rejects_bad_type():
    with pytest.raises(TypeError, match="free_ic must be a FreeICConfig or None"):
        _free_ic_fields(["some_species"])


# ---------- _laplace_covariance: guards ----------


def test_laplace_covariance_rejects_out_of_range_eig_keep():
    H = np.eye(2)
    with pytest.raises(ValueError, match=r"eig_keep must be in \[0, 1\)"):
        _laplace_covariance(H, ridge=1e-6, eig_keep=1.0)


def test_laplace_covariance_rejects_non_pd_hessian():
    # A zero Hessian ridged by 0 has w_max == 0 -> not positive-definite.
    H = np.zeros((2, 2))
    with pytest.raises(ValueError, match="not finite / positive-definite"):
        _laplace_covariance(H, ridge=0.0, eig_keep=1e-2)


# ---------- _resolve_problem: further validation ----------


def test_resolve_problem_rejects_mismatched_sigma_list(base):
    m, C0, obs, t_obs, free = base
    # Multi-dataset with a sigma list of the wrong length.
    with pytest.raises(ValueError, match="sigma list has 1 entries but there are 2"):
        _resolve(
            m,
            [C0, C0],
            [obs, obs],
            [t_obs, t_obs],
            free,
            loss="wmse",
            sigma=[jnp.ones_like(obs)],
        )


def test_resolve_problem_rejects_2d_tobs(base):
    m, C0, obs, _t, free = base
    with pytest.raises(ValueError, match="t_obs must be a non-empty 1-D array"):
        _resolve(m, C0, obs, jnp.zeros((3, 1)), free)


def test_resolve_problem_rejects_negative_tobs(base):
    m, C0, obs, _t, free = base
    with pytest.raises(ValueError, match="t_obs must be non-negative"):
        _resolve(m, C0, obs, jnp.array([-1.0, 1.0, 2.0]), free)


def test_resolve_problem_rejects_obs_row_mismatch(base):
    m, C0, _obs, t_obs, free = base
    # t_obs has 3 entries; give observations with 2 rows.
    bad_obs = jnp.zeros((2, m.n_species))
    with pytest.raises(ValueError, match="observations has 2 rows but t_obs"):
        _resolve(m, C0, bad_obs, t_obs, free)


def test_resolve_problem_rejects_obs_col_mismatch(base):
    m, C0, _obs, t_obs, free = base
    # Ask for a single observed species but hand over all-species columns.
    bad_obs = jnp.zeros((3, m.n_species))
    with pytest.raises(ValueError, match="columns but 1 species were specified"):
        _resolve(m, C0, bad_obs, t_obs, free, observed_species=[m.species[0]])


# ---------- calibrate(): top-level argument guards (no solve) ----------


def _batch_reactor(model):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    return aquakin.BatchReactor(model, conditions)


def _cal_args(model):
    C0 = jnp.array([1.0, 0.0])
    t_obs = jnp.array([0.0, 1.0, 2.0])
    obs = jnp.zeros((3, model.n_species))
    return C0, obs, t_obs


def test_calibrate_forward_plus_stable_is_rejected(simple_model):
    reactor = _batch_reactor(simple_model)
    C0, obs, t_obs = _cal_args(simple_model)
    with pytest.raises(ValueError, match="incompatible: the stable discrete adjoint"):
        calibrate(
            reactor,
            C0,
            obs,
            t_obs,
            [simple_model.parameters[0]],
            diff=DifferentiationConfig(mode="forward", method="stable"),
        )


def test_calibrate_stable_adjoint_requires_batch_reactor(simple_model):
    # A reactor exposing .model but no .conditions is not a batch reactor;
    # the stable adjoint is implemented only for batch reactors.
    class _NoConditions:
        model = simple_model

    C0, obs, t_obs = _cal_args(simple_model)
    with pytest.raises(ValueError, match="gradient='stable_adjoint' is implemented for batch"):
        calibrate(
            _NoConditions(),
            C0,
            obs,
            t_obs,
            [simple_model.parameters[0]],
            diff=DifferentiationConfig(mode="reverse", method="stable"),
        )


# ---------- CalibrationResult.predictive_band: degenerate posterior ----------


def _minimal_result(model, posterior_cov):
    name = model.parameters[0]
    return CalibrationResult(
        params=model.default_parameters(),
        params_named={name: float(model.default_parameters()[model.param_index[name]])},
        loss=0.0,
        converged=True,
        message="ok",
        n_iter=0,
        parameter_names=[name],
        transforms=["none"],
        posterior_cov=posterior_cov,
    )


def test_predictive_band_requires_laplace(simple_model):
    result = _minimal_result(simple_model, posterior_cov=None)
    reactor = _batch_reactor(simple_model)
    with pytest.raises(ValueError, match="predictive_band requires a Laplace posterior"):
        result.predictive_band(reactor, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))


def test_predictive_band_rejects_degenerate_covariance(simple_model):
    # A zero posterior covariance has no positive-variance directions; the
    # guard fires before any solve is attempted.
    result = _minimal_result(simple_model, posterior_cov=np.zeros((1, 1)))
    reactor = _batch_reactor(simple_model)
    with pytest.raises(ValueError, match="no positive-variance directions"):
        result.predictive_band(reactor, jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
