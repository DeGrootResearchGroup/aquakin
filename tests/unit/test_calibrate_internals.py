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

from aquakin.integrate.calibrate import (
    _CalibrationProblem,
    _FitConfig,
    _forward_jac,
    _optimizer_bounds,
    _resolve_problem,
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
