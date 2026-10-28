"""Unit tests for :func:`calibrate`'s extracted building blocks.

The god-function refactor split ``calibrate`` into independently testable pieces:
:func:`_resolve_problem` (validation + coercion), :class:`_CalibrationProblem`
(the resolved static problem), :func:`_optimizer_bounds`, and :func:`_forward_jac`
(the AD-direction rule). These exercise them directly -- no fit, no solve -- which
the previous single-closure structure could not. Behaviour equivalence to the old
monolith is covered by the full-fit tests in
``tests/integration/test_calibrate.py``; here we pin the units.
"""

import dataclasses
import sys

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate._common import DifferentiationConfig
from aquakin.integrate.calibrate import (
    _DEFAULT_IC_BOUNDS,
    CalibrationResult,
    _at_bound_named,
    _at_bound_sides,
    _build_loss,
    _build_residual,
    _CalibrationProblem,
    _check_start_gradient,
    _data_term,
    _FitConfig,
    _forward_jac,
    _free_ic_fields,
    _ic_coordinate_names,
    _is_solve_failure,
    _joint_names,
    _laplace_covariance,
    _laplace_posterior,
    _missing_mask,
    _ObjectiveBundle,
    _optimizer_bounds,
    _per_species_ic_bounds,
    _ReactorForwardModel,
    _resolve_laplace,
    _resolve_problem,
    _run_multistart,
    _simplex_fraction_cov,
    _simplex_pools,
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


def _key_variants(m, C0, obs, t_obs, free):
    """Problems that differ from the base in one thing the compiled objective
    bakes in, by name. Each must key differently from the base and from every
    other variant."""
    name = free[0]
    sp = [m.species[1]]
    obs1 = obs[:, 1:]
    nll = dict(loss="nll", sigma=0.1)
    return {
        "base": _resolve(m, C0, obs1, t_obs, free, observed_species=sp),
        "observations": _resolve(m, C0, obs1 + 0.5, t_obs, free, observed_species=sp),
        "one observation": _resolve(
            m, C0, obs1.at[1, 0].add(1e-9), t_obs, free, observed_species=sp
        ),
        "t_obs values": _resolve(m, C0, obs1, t_obs + 0.25, free, observed_species=sp),
        "sigma": _resolve(m, C0, obs1, t_obs, free, observed_species=sp, **nll),
        # Same sigma, different loss type.
        "loss": _resolve(m, C0, obs1, t_obs, free, observed_species=sp, loss="wmse", sigma=0.1),
        "sigma value": _resolve(
            m, C0, obs1, t_obs, free, observed_species=sp, loss="nll", sigma=0.2
        ),
        # The concentrated likelihood compiles a different objective and
        # residual from any fixed sigma.
        "sigma estimate": _resolve(
            m, C0, obs1, t_obs, free, observed_species=sp, loss="nll", sigma="estimate"
        ),
        "sigma per point": _resolve(
            m,
            C0,
            obs1,
            t_obs,
            free,
            observed_species=sp,
            loss="nll",
            sigma=jnp.array([[0.1], [0.2], [0.1]]),
        ),
        "transform": _resolve(
            m, C0, obs1, t_obs, free, observed_species=sp, transforms={name: "positive_log"}
        ),
        "observed species": _resolve(m, C0, obs1, t_obs, free, observed_species=[m.species[0]]),
        "observation_fn": _resolve(m, C0, obs1, t_obs, free, observation_fn=lambda C, p: C[1]),
        "prior": _resolve(
            m,
            C0,
            obs1,
            t_obs,
            free,
            observed_species=sp,
            priors={name: (0.1, 0.05)},
        ),
        "prior std": _resolve(
            m,
            C0,
            obs1,
            t_obs,
            free,
            observed_species=sp,
            priors={name: (0.1, 0.06)},
        ),
        "free initial pool": _resolve(
            m, C0, obs1, t_obs, free, observed_species=sp, free_ic=[m.species[0]]
        ),
    }


def test_struct_key_changes_with_everything_the_objective_bakes_in(base):
    """Every variant keys differently from the base and from every other
    variant, so a shared compiled-objective cache cannot hand one of them a
    program compiled for another."""
    variants = _key_variants(*base)
    keys = {label: prob.struct_key("jax_adjoint") for label, prob in variants.items()}
    for label, k in keys.items():
        hash(k)  # usable as a dict key
        if label != "base":
            assert k != keys["base"], label
    assert len(set(keys.values())) == len(keys)


def test_struct_key_equal_for_equal_inputs(base):
    """Equal content keys equally, even through fresh array objects and other
    container types, so a repeat fit reuses the compiled program."""
    m, C0, obs, t_obs, free = base
    a = _resolve(m, C0, obs, t_obs, free, loss="nll", sigma=0.1)
    b = _resolve(
        m,
        jnp.array(np.asarray(C0)),
        np.array(obs).copy(),
        list(np.asarray(t_obs)),
        list(free),
        loss="nll",
        sigma=jnp.asarray(0.1),
    )
    assert a.struct_key("jax_adjoint") == b.struct_key("jax_adjoint")


def test_struct_key_ignores_runtime_arguments(base):
    """The initial parameters, the initial state and the rate bounds are not
    baked into the compiled program (the first two are runtime arguments, the
    bounds go only to SciPy), so changing them keeps the key: a
    profile_likelihood sweep reuses one program."""
    m, C0, obs, t_obs, free = base
    k = _resolve(m, C0, obs, t_obs, free).struct_key("jax_adjoint")
    moved = [
        _resolve(m, C0, obs, t_obs, free, initial_params=m.default_parameters().at[0].set(3.0)),
        _resolve(m, C0.at[0].set(0.4), obs, t_obs, free),
        _resolve(m, C0, obs, t_obs, free, bounds={free[0]: (0.05, 0.5)}),
        _resolve(m, C0, obs, t_obs, free, use_model_bounds=False),
        _resolve(m, C0, obs, t_obs, free, param_halfwidth=1.0),
    ]
    for prob in moved:
        assert prob.struct_key("jax_adjoint") == k


# ---------- _optimizer_bounds ----------


def test_optimizer_bounds_unbounded_without_model_bounds(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free, use_model_bounds=False)
    bounds, lb, ub, ls_bounds, has = _optimizer_bounds(prob, prob.rate_theta0())
    assert bounds is None and not has
    assert not np.isfinite(lb[0]) and not np.isfinite(ub[0])


def test_optimizer_bounds_use_declared_bounds_by_default(base):
    """The fixture declares k in [1e-3, 10]; with the model's positive_log
    transform the optimiser box is [log 1e-3, log 10]."""
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free, transforms={free[0]: "positive_log"})
    bounds, lb, ub, _ls, has = _optimizer_bounds(prob, prob.rate_theta0())
    assert has and bounds is not None
    assert lb[0] == pytest.approx(np.log(1e-3))
    assert ub[0] == pytest.approx(np.log(10.0))


def test_optimizer_bounds_explicit_override_and_open_side(base):
    m, C0, obs, t_obs, free = base
    tr = {free[0]: "positive_log"}
    prob = _resolve(m, C0, obs, t_obs, free, transforms=tr, bounds={free[0]: (0.05, None)})
    _b, lb, ub, _ls, _has = _optimizer_bounds(prob, prob.rate_theta0())
    assert lb[0] == pytest.approx(np.log(0.05))
    assert ub[0] == np.inf
    # lo <= 0 is the edge of positive_log's domain: open, not log(0).
    prob = _resolve(m, C0, obs, t_obs, free, transforms=tr, bounds={free[0]: (0.0, 2.0)})
    _b, lb, ub, _ls, _has = _optimizer_bounds(prob, prob.rate_theta0())
    assert lb[0] == -np.inf and ub[0] == pytest.approx(np.log(2.0))


def test_optimizer_bounds_logit_and_identity_mapping(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(
        m, C0, obs, t_obs, free, transforms={free[0]: "logit"}, bounds={free[0]: (0.05, 0.5)}
    )
    _b, lb, ub, _ls, _has = _optimizer_bounds(prob, prob.rate_theta0())
    assert lb[0] == pytest.approx(np.log(0.05 / 0.95))
    assert ub[0] == pytest.approx(0.0)
    prob = _resolve(
        m, C0, obs, t_obs, free, transforms={free[0]: "none"}, bounds={free[0]: (0.05, 0.5)}
    )
    _b, lb, ub, _ls, _has = _optimizer_bounds(prob, prob.rate_theta0())
    assert (lb[0], ub[0]) == (pytest.approx(0.05), pytest.approx(0.5))


def test_optimizer_bounds_non_finite_side_is_open(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(
        m,
        C0,
        obs,
        t_obs,
        free,
        transforms={free[0]: "positive_log"},
        bounds={free[0]: (0.05, np.inf)},
    )
    _b, lb, ub, _ls, _has = _optimizer_bounds(prob, prob.rate_theta0())
    assert lb[0] == pytest.approx(np.log(0.05))
    assert ub[0] == np.inf


def test_optimizer_bounds_logit_domain_edges_are_open(base):
    """[0, 1] is the whole of logit's domain, so both sides are open and the
    explicit entry leaves the rate unbounded (it also replaces the declared
    [1e-3, 10])."""
    m, C0, obs, t_obs, free = base
    prob = _resolve(
        m, C0, obs, t_obs, free, transforms={free[0]: "logit"}, bounds={free[0]: (0.0, 1.0)}
    )
    assert prob.rate_bounds is None
    bounds, lb, ub, _ls, has = _optimizer_bounds(prob, prob.rate_theta0())
    assert bounds is None and not has
    assert lb[0] == -np.inf and ub[0] == np.inf
    # One side inside the domain: only the other is open.
    prob = _resolve(
        m, C0, obs, t_obs, free, transforms={free[0]: "logit"}, bounds={free[0]: (0.2, 1.5)}
    )
    _b, lb, ub, _ls, _has = _optimizer_bounds(prob, prob.rate_theta0())
    assert lb[0] == pytest.approx(np.log(0.2 / 0.8))
    assert ub[0] == np.inf


def test_optimizer_bounds_free_ic_log_bounded(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(
        m,
        C0,
        obs,
        t_obs,
        free,
        free_ic=[m.species[0]],
        ic_bounds=(1e-2, 10.0),
        use_model_bounds=False,
    )
    bounds, lb, ub, _ls, has = _optimizer_bounds(prob, prob.rate_theta0())
    assert has
    # One rate dim (unbounded) + one free-IC dim (log-bounded by ic_bounds).
    assert lb[0] == -np.inf and ub[0] == np.inf
    assert lb[-1] == pytest.approx(np.log(1e-2))
    assert ub[-1] == pytest.approx(np.log(10.0))


def test_optimizer_bounds_halfwidth_intersects_absolute_bounds(base):
    """The halfwidth box is intersected with the absolute bounds: the start
    k=0.1 with halfwidth log(1000) would reach 100, but the declared upper
    bound 10 wins; the lower side is the halfwidth's 1e-4 -> declared 1e-3."""
    m, C0, obs, t_obs, free = base
    prob = _resolve(
        m,
        C0,
        obs,
        t_obs,
        free,
        transforms={free[0]: "positive_log"},
        param_halfwidth=float(np.log(1000.0)),
    )
    _b, lb, ub, _ls, _has = _optimizer_bounds(prob, prob.rate_theta0())
    assert lb[0] == pytest.approx(np.log(1e-3))
    assert ub[0] == pytest.approx(np.log(10.0))
    prob = _resolve(
        m, C0, obs, t_obs, free, transforms={free[0]: "positive_log"}, param_halfwidth=0.5
    )
    _b, lb, ub, _ls, _has = _optimizer_bounds(prob, prob.rate_theta0())
    assert lb[0] == pytest.approx(np.log(0.1) - 0.5)
    assert ub[0] == pytest.approx(np.log(0.1) + 0.5)


def test_resolve_problem_rejects_bad_bounds(base):
    m, C0, obs, t_obs, free = base
    with pytest.raises(KeyError, match="Unknown parameter 'nope'"):
        _resolve(m, C0, obs, t_obs, free, bounds={"nope": (0.0, 1.0)})
    with pytest.raises(ValueError, match="lo < hi"):
        _resolve(m, C0, obs, t_obs, free, bounds={free[0]: (1.0, 1.0)})
    with pytest.raises(ValueError, match=r"\(lo, hi\) pair"):
        _resolve(m, C0, obs, t_obs, free, bounds={free[0]: (1.0,)})


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


def test_laplace_covariance_rejects_non_pd_hessian():
    # A zero Hessian has no positive eigenvalue -> not positive-definite.
    H = np.zeros((2, 2))
    with pytest.raises(ValueError, match="not finite / positive-definite"):
        _laplace_covariance(H, ridge=1e-6)


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


# ---------- solve failures: _is_solve_failure / _run_multistart ----------


class EquinoxRuntimeError(RuntimeError):
    """Stand-in carrying the class name of equinox's runtime error."""


_CALLBACK_MSG = "INTERNAL: CpuCallback error calling callback: non-finite output"


@pytest.mark.parametrize(
    "exc,expected",
    [
        (EquinoxRuntimeError("linear solve failed"), True),
        (RuntimeError(_CALLBACK_MSG), True),
        (ValueError(_CALLBACK_MSG), True),
        (ValueError("bad shape"), False),
        (RuntimeError("something else"), False),
        (TypeError(_CALLBACK_MSG), False),
    ],
)
def test_is_solve_failure(exc, expected):
    assert _is_solve_failure(exc) is expected


def _quadratic_bundle(fails):
    """L-BFGS-B bundle for the loss ``sum(x**2)`` whose evaluation raises
    ``fails(x)`` (an exception, or ``None`` to succeed)."""

    def value_and_grad(x):
        exc = fails(np.asarray(x))
        if exc is not None:
            raise exc
        return jnp.sum(x**2), 2.0 * x

    return _ObjectiveBundle(value_and_grad, None, None, False)


def _ms_cfg(n_starts, optimizer="lbfgsb"):
    return dataclasses.replace(
        _cfg("auto", "jax_adjoint"),
        optimizer=optimizer,
        max_iter=50,
        n_starts=n_starts,
        jitter_schedule=(1.0,),
    )


_NO_BOUNDS = (None, None, None, (-np.inf, np.inf), False)
_THETA0 = jnp.asarray([2.0])


def test_run_multistart_uses_jittered_start_when_first_fails():
    """A failing unperturbed start is skipped and a surviving jittered start is
    returned, with a warning naming start 0."""
    calls = []

    def fails(x):
        calls.append(x.copy())
        return RuntimeError(_CALLBACK_MSG) if len(calls) == 1 else None

    with pytest.warns(RuntimeWarning, match=r"start index 0;"):
        out = _run_multistart(_ms_cfg(2), _quadratic_bundle(fails), _THETA0, _NO_BOUNDS)
    assert out.fun == pytest.approx(0.0, abs=1e-10)


def test_run_multistart_single_start_failure_raises():
    """With ``n_starts=1`` a failed run raises the actionable error, chained."""
    bundle = _quadratic_bundle(lambda x: ValueError(_CALLBACK_MSG))
    with pytest.raises(RuntimeError, match="in the optimiser run") as info:
        _run_multistart(_ms_cfg(1), bundle, _THETA0, _NO_BOUNDS)
    assert "param_halfwidth" in str(info.value)
    assert isinstance(info.value.__cause__, ValueError)


def test_run_multistart_propagates_non_solve_errors():
    """An exception that is not a solve failure is not treated as a failed start."""
    bundle = _quadratic_bundle(lambda x: TypeError("bug"))
    with pytest.raises(TypeError, match="bug"):
        _run_multistart(_ms_cfg(3), bundle, _THETA0, _NO_BOUNDS)


def test_check_start_gradient_wraps_solve_failure():
    bundle = _quadratic_bundle(lambda x: RuntimeError(_CALLBACK_MSG))
    with pytest.raises(RuntimeError, match="at the starting guess"):
        _check_start_gradient(_ms_cfg(1), bundle, _THETA0)


def test_check_start_gradient_propagates_non_solve_errors():
    bundle = _quadratic_bundle(lambda x: TypeError("bug"))
    with pytest.raises(TypeError, match="bug"):
        _check_start_gradient(_ms_cfg(1), bundle, _THETA0)


# ---------- _laplace_posterior: at-bound mask ----------


def test_laplace_posterior_at_bound_mask(simple_model):
    """No mask (Plant.calibrate's call) and an all-False mask give the same
    posterior; masking the only rate holds it fixed: nan std, zero covariance."""
    m = simple_model
    reactor = _batch_reactor(m)
    C0 = jnp.array([1.0, 0.0])
    t_obs = jnp.array([1.0, 2.0, 4.0])
    obs = reactor.solve(C0, params=m.default_parameters(), t_span=(0.0, 4.0), t_eval=t_obs).C
    prob = _resolve(m, C0, obs, t_obs, [m.parameters[0]], use_model_bounds=False)
    cfg = dataclasses.replace(_cfg("auto", "jax_adjoint"), laplace=True)
    fm = _ReactorForwardModel(reactor, "jax_adjoint", 1, False)
    theta = prob.rate_theta0()
    no_mask = _laplace_posterior(prob, fm, cfg, theta, jnp.zeros(0))
    all_free = _laplace_posterior(
        prob, fm, cfg, theta, jnp.zeros(0), at_bound_mask=np.array([False])
    )
    assert np.isfinite(no_mask.posterior_std_unconstrained[0])
    np.testing.assert_array_equal(no_mask.posterior_cov, all_free.posterior_cov)
    held = _laplace_posterior(prob, fm, cfg, theta, jnp.zeros(0), at_bound_mask=np.array([True]))
    assert np.isnan(held.posterior_std_unconstrained[0])
    assert np.all(np.asarray(held.posterior_cov) == 0.0)


# ---------- FreeICConfig: per-species bounds and the fixed-total split ----------


def test_per_species_ic_bounds_forms():
    """A shared pair applies to every pool; a dict sets named pools and leaves
    the rest at the default."""
    lo, hi = _per_species_ic_bounds(["S", "X"], (0.1, 2.0))
    np.testing.assert_array_equal(lo, [0.1, 0.1])
    np.testing.assert_array_equal(hi, [2.0, 2.0])
    lo, hi = _per_species_ic_bounds(["S", "X"], {"X": (5.0, 50.0)})
    np.testing.assert_array_equal(lo, [_DEFAULT_IC_BOUNDS[0], 5.0])
    np.testing.assert_array_equal(hi, [_DEFAULT_IC_BOUNDS[1], 50.0])


@pytest.mark.parametrize(
    "bounds, exc, match",
    [
        ({"Q": (1.0, 2.0)}, KeyError, "not fitted"),
        ({"S": (2.0, 1.0)}, ValueError, "0 < lo < hi"),
        ({"S": (0.0, 1.0)}, ValueError, "0 < lo < hi"),
        ({"S": (-1.0, 1.0)}, ValueError, "0 < lo < hi"),
        ({"S": (1.0, 1.0)}, ValueError, "0 < lo < hi"),
        ({"S": (float("nan"), 1.0)}, ValueError, "0 < lo < hi"),
        ({"S": 5.0}, ValueError, r"\(lo, hi\) pair"),
        ({"S": (1.0, 2.0, 3.0)}, ValueError, r"\(lo, hi\) pair"),
        ((1.0,), ValueError, r"\(lo, hi\) pair"),
    ],
)
def test_per_species_ic_bounds_rejects_malformed(bounds, exc, match):
    with pytest.raises(exc, match=match):
        _per_species_ic_bounds(["S", "X"], bounds)


def test_resolve_problem_per_species_ic_bounds(base):
    """Per-species bounds become per-pool arrays; each pool's start is clipped
    into its own box (A starts at 1.0 above its cap 0.5, B at 0.0 below its
    floor 0.2), and the optimiser box is the log of those arrays, tiled over
    datasets."""
    m, C0, obs, t_obs, free = base
    prob = _resolve(
        m,
        [C0, C0],
        [obs, obs],
        [t_obs, t_obs],
        free,
        free_ic=["A", "B"],
        ic_bounds={"A": (0.01, 0.5), "B": (0.2, 3.0)},
        use_model_bounds=False,
    )
    np.testing.assert_array_equal(prob.ic_bounds[0], [0.01, 0.2])
    np.testing.assert_array_equal(prob.ic_bounds[1], [0.5, 3.0])
    np.testing.assert_allclose(prob.ic_center_full, np.log([0.5, 0.2, 0.5, 0.2]))
    _b, lb, ub, _ls, has = _optimizer_bounds(prob, prob.rate_theta0())
    assert has
    np.testing.assert_allclose(lb[1:], np.log([0.01, 0.2, 0.01, 0.2]))
    np.testing.assert_allclose(ub[1:], np.log([0.5, 3.0, 0.5, 3.0]))


def test_at_bound_names_a_pool_on_its_own_bound(base):
    """A pool sitting on its own per-species bound is flagged by name; the
    other pool, inside a different box at the same value, is not."""
    m, C0, obs, t_obs, free = base
    prob = _resolve(
        m,
        C0,
        obs,
        t_obs,
        free,
        free_ic=["A", "B"],
        ic_bounds={"A": (0.01, 0.5), "B": (0.01, 3.0)},
        use_model_bounds=False,
    )
    opt_bounds = _optimizer_bounds(prob, prob.rate_theta0())
    theta = np.concatenate([np.asarray(prob.rate_theta0()), np.log([0.5, 0.5])])
    sides = _at_bound_sides(theta, opt_bounds)
    assert _at_bound_named(prob, sides) == {"ic[0].A": "upper"}
    theta[2] = np.log(0.01)
    assert _at_bound_named(prob, _at_bound_sides(theta, opt_bounds)) == {
        "ic[0].A": "upper",
        "ic[0].B": "lower",
    }


def test_resolve_problem_fixed_total(base):
    """The split is centred on the log-ratios of the C0 pools; its coordinates
    are unbounded, so with no rate bounds nothing is bounded and nothing can
    be flagged, and the pools always sum to each dataset's total."""
    m, _C0, obs, t_obs, free = base
    C0a, C0b = jnp.array([0.6, 0.4]), jnp.array([1.0, 3.0])
    prob = _resolve(
        m,
        [C0a, C0b],
        [obs, obs],
        [t_obs, t_obs],
        free,
        free_ic=["A", "B"],
        ic_fixed_total=True,
        use_model_bounds=False,
    )
    assert prob.ic_fixed_total and prob.m_ic == 2 and prob.m_ic_theta == 1
    np.testing.assert_allclose(prob.ic_center_full, [np.log(0.4 / 0.6), np.log(3.0)])
    bounds, lb, ub, _ls, has = _optimizer_bounds(prob, prob.rate_theta0())
    assert not has and bounds is None
    assert np.all(np.isinf(lb)) and np.all(np.isinf(ub))
    theta = np.concatenate([np.asarray(prob.rate_theta0()), [50.0, -50.0]])
    assert _at_bound_named(prob, _at_bound_sides(theta, (bounds, lb, ub, _ls, has))) == {}
    assert _joint_names(prob) == [free[0], "ic[0].B:A", "ic[1].B:A"]
    # The start reproduces the C0 pools; each dataset keeps its own total.
    for k, C0_k in enumerate((C0a, C0b)):
        z = prob.ic_center_full[k : k + 1]
        np.testing.assert_allclose(prob.ic_pools(z, C0_k), C0_k, rtol=1e-12)
        pools = prob.ic_pools(jnp.array([7.3]), C0_k)
        assert float(jnp.sum(pools)) == pytest.approx(float(jnp.sum(C0_k)), rel=1e-12)


def test_fixed_total_with_rate_bounds_keeps_split_open(base):
    """Rate bounds stay finite (and flaggable) next to the unbounded split."""
    m, _C0, obs, t_obs, free = base
    prob = _resolve(
        m,
        jnp.array([0.6, 0.4]),
        obs,
        t_obs,
        free,
        free_ic=["A", "B"],
        ic_fixed_total=True,
        transforms={free[0]: "positive_log"},
    )
    opt_bounds = _optimizer_bounds(prob, prob.rate_theta0())
    _b, lb, ub, _ls, has = opt_bounds
    assert has
    assert lb[0] == pytest.approx(np.log(1e-3)) and ub[0] == pytest.approx(np.log(10.0))
    assert lb[1] == -np.inf and ub[1] == np.inf
    sides = _at_bound_sides(np.array([np.log(10.0), 0.0]), opt_bounds)
    assert _at_bound_named(prob, sides) == {free[0]: "upper"}


def test_resolve_problem_fixed_total_validation(base):
    m, _C0, obs, t_obs, free = base
    with pytest.raises(ValueError, match="takes no bounds"):
        _resolve(
            m,
            jnp.array([0.6, 0.4]),
            obs,
            t_obs,
            free,
            free_ic=["A", "B"],
            ic_fixed_total=True,
            ic_bounds={"A": (0.1, 1.0)},
        )
    with pytest.raises(ValueError, match=r"dataset 0.*must be positive in C0"):
        _resolve(
            m, jnp.array([1.0, 0.0]), obs, t_obs, free, free_ic=["A", "B"], ic_fixed_total=True
        )
    # fixed_total without free pools is a no-op.
    prob = _resolve(m, jnp.array([1.0, 0.0]), obs, t_obs, free, ic_fixed_total=True)
    assert not prob.ic_fixed_total and prob.m_ic_theta == 0


def test_fixed_total_single_pool_is_held_at_its_total(base):
    """One pool in a fixed total has no split to fit: no coordinates, and the
    pool is its C0 value."""
    m, _C0, obs, t_obs, free = base
    C0 = jnp.array([0.7, 0.0])
    prob = _resolve(m, C0, obs, t_obs, free, free_ic=["A"], ic_fixed_total=True)
    assert prob.m_ic == 1 and prob.m_ic_theta == 0
    assert prob.ic_center_full.shape == (0,)
    np.testing.assert_allclose(prob.ic_pools(jnp.zeros(0), C0), [0.7])
    assert _joint_names(prob) == [free[0]]


def test_ic_pools_without_fixed_total_is_exp(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free, free_ic=["A", "B"])
    assert prob.m_ic_theta == 2
    np.testing.assert_allclose(prob.ic_pools(jnp.log(jnp.array([2.0, 3.0])), C0), [2.0, 3.0])
    assert _joint_names(prob) == [free[0], "ic[0].A", "ic[0].B"]


def test_simplex_pools_extremes_stay_positive_and_conserve_total():
    """Large log-ratios saturate without overflow: the pools stay finite,
    non-negative and sum to the total."""
    for z in ([0.0, 0.0], [800.0, -800.0], [-800.0, -800.0], [1e-12, 40.0]):
        pools = np.asarray(_simplex_pools(jnp.asarray(z), 4.0))
        assert np.all(np.isfinite(pools)) and np.all(pools >= 0.0)
        assert pools.sum() == pytest.approx(4.0, rel=1e-12)
    np.testing.assert_allclose(_simplex_pools(jnp.zeros(2), 3.0), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(_simplex_pools(jnp.zeros(0), 3.0), [3.0])


def test_simplex_fraction_cov_is_delta_method():
    """The fraction covariance is J cov_z J^T with J the softmax Jacobian: it
    matches a finite-difference J, and since the fractions sum to 1 every
    row sums to 0."""
    z = np.array([0.4, -1.1])
    cov_z = np.array([[0.04, 0.01], [0.01, 0.09]])
    cov_f = _simplex_fraction_cov(z, cov_z)

    def frac(zz):
        return np.asarray(_simplex_pools(jnp.asarray(zz), 1.0))

    h = 1e-6
    J = np.stack([(frac(z + h * e) - frac(z - h * e)) / (2 * h) for e in np.eye(2)], axis=1)
    np.testing.assert_allclose(cov_f, J @ cov_z @ J.T, rtol=1e-6, atol=1e-14)
    np.testing.assert_allclose(cov_f.sum(axis=1), 0.0, atol=1e-15)
    # One pool: no coordinates, zero variance.
    np.testing.assert_array_equal(_simplex_fraction_cov(np.zeros(0), np.zeros((0, 0))), [[0.0]])


def test_ic_coordinate_names():
    assert _ic_coordinate_names(["S", "X", "P"], 2, False) == [
        "ic[0].S",
        "ic[0].X",
        "ic[0].P",
        "ic[1].S",
        "ic[1].X",
        "ic[1].P",
    ]
    assert _ic_coordinate_names(["S", "X", "P"], 2, True) == [
        "ic[0].X:S",
        "ic[0].P:S",
        "ic[1].X:S",
        "ic[1].P:S",
    ]


def test_free_ic_fields_extended_passes_extensions_through():
    cfg = aquakin.FreeICConfig(["S", "X"], bounds={"S": (1.0, 2.0)}, fixed_total=False)
    species, bounds, prior = _free_ic_fields(cfg, extended=True)
    assert species == ["S", "X"] and bounds == {"S": (1.0, 2.0)} and prior is None


def test_resolve_problem_rejects_duplicate_free_ic(base):
    m, C0, obs, t_obs, free = base
    with pytest.raises(ValueError, match="distinct"):
        _resolve(m, C0, obs, t_obs, free, free_ic=["A", "A"])


# ---------- sigma="estimate" (concentrated likelihood): guards ----------


@pytest.mark.parametrize("build", [_build_loss, _build_residual])
def test_build_sigma_estimate_requires_nll(build):
    with pytest.raises(ValueError, match='sigma="estimate" requires loss="nll"'):
        build("mse", jnp.zeros((3, 1)), "estimate")


def test_resolve_problem_rejects_unknown_sigma_string(base):
    m, C0, obs, t_obs, free = base
    with pytest.raises(ValueError, match='sigma must be an array, None or "estimate"'):
        _resolve(m, C0, obs, t_obs, free, loss="nll", sigma="bogus")


def test_resolve_problem_rejects_estimate_in_sigma_list(base):
    """The concentrated likelihood pools every dataset's residuals, so it cannot
    be asked for per dataset."""
    m, C0, obs, t_obs, free = base
    with pytest.raises(ValueError, match="applies to every dataset"):
        _resolve(m, [C0, C0], [obs, obs], [t_obs, t_obs], free, loss="nll", sigma=["estimate", 0.1])


def test_laplace_posterior_sigma_estimate_needs_noise_estimate(base):
    """The plug-in Gauss-Newton Laplace of the concentrated likelihood needs the
    noise estimate at the MAP; it is an error to omit it."""
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, C0, obs, t_obs, free, loss="nll", sigma="estimate")
    cfg = dataclasses.replace(_cfg("auto", "jax_adjoint"), laplace_method="gauss_newton")
    fm = _ReactorForwardModel(_batch_reactor(m), "jax_adjoint", 1, False)
    with pytest.raises(ValueError, match="needs the noise estimate"):
        _laplace_posterior(prob, fm, cfg, prob.rate_theta0(), jnp.zeros(0))


def test_run_multistart_reports_unconverged_noise_reestimation(monkeypatch):
    """Gauss-Newton under the concentrated likelihood alternates a least-squares
    solve with re-estimating sigma; if sigma has not settled within the
    iteration cap the run is reported as not converged."""
    monkeypatch.setattr(sys.modules[_run_multistart.__module__], "_IRLS_MAX_ITER", 3)
    box = [np.ones(1)]
    calls = [0]

    def sigma_hat_of(x):
        calls[0] += 1  # a noise estimate that never settles
        return np.asarray([float(calls[0])])

    def set_sigma_hat(v):
        box[0] = np.asarray(v)

    bundle = _ObjectiveBundle(
        lambda x: (jnp.sum((x - 1.0) ** 2), 2.0 * (x - 1.0)),
        lambda x: (x - 1.0) / box[0],
        lambda x: jnp.eye(1) / box[0],
        False,
        sigma_hat_of,
        set_sigma_hat,
    )
    cfg = dataclasses.replace(_ms_cfg(1, optimizer="gauss_newton"), max_iter=20)
    out = _run_multistart(cfg, bundle, _THETA0, _NO_BOUNDS)
    assert not out.success
    assert out.message == "noise re-estimation did not converge"
    assert out.x[0] == pytest.approx(1.0)


def test_resolve_problem_accepts_numeric_sigma_list(base):
    m, C0, obs, t_obs, free = base
    prob = _resolve(m, [C0, C0], [obs, obs], [t_obs, t_obs], free, loss="nll", sigma=[0.1, 0.2])
    assert prob.n_datasets == 2
    assert not prob.sigma_estimate


# ---------- missing (NaN) observations with sigma="estimate" ----------


def _obs_with_gap(base):
    """Two observed outputs on three times; output 1 is missing at t = 1."""
    m, C0, _obs, t_obs, free = base
    obs = jnp.array([[0.9, 0.1], [0.5, jnp.nan], [0.2, 0.8]])
    return m, C0, obs, t_obs, free


def test_sigma_estimate_counts_only_observed_entries(base):
    """N_j is the number of observed entries of output j, pooled over the
    datasets, and the NaN entries add nothing to the sum of squares or the raw
    residual."""
    m, C0, obs, t_obs, free = _obs_with_gap(base)
    prob = _resolve(m, [C0, C0], [obs, obs], [t_obs, t_obs], free, loss="nll", sigma="estimate")
    np.testing.assert_array_equal(prob.n_obs_per_output, [6.0, 4.0])

    pred = jnp.array([[1.0, 0.0], [0.4, 123.0], [0.2, 1.0]])
    _t, _ts, sse_fn, resid_fn = prob.dataset_static[0]
    np.testing.assert_allclose(sse_fn(pred), [0.01 + 0.01, 0.01 + 0.04])
    r = np.asarray(resid_fn(pred))
    assert r[1, 1] == 0.0 and np.all(np.isfinite(r))

    # The concentrated objective is finite and equals sum_j (N_j/2) log(SSE_j/N_j) + N/2.
    sse = np.array([0.04, 0.10])
    n = np.array([6.0, 4.0])
    expected = 0.5 * np.sum(n * np.log(sse / n)) + 0.5 * np.sum(n)
    assert float(_data_term(prob, (pred, pred))) == pytest.approx(expected, rel=1e-12)


def test_sigma_estimate_rejects_an_output_never_observed(base):
    m, C0, _obs, t_obs, free = base
    obs = jnp.array([[0.9, jnp.nan], [0.5, jnp.nan], [0.2, jnp.nan]])
    with pytest.raises(ValueError, match=r"output\(s\) \[1\] are missing"):
        _resolve(m, C0, obs, t_obs, free, loss="nll", sigma="estimate")


# ---------- missing (NaN) observations: masked losses and guards ----------

_OBS_GAP = jnp.array([[1.0, 2.0], [jnp.nan, 4.0]])
_PRED_GAP = jnp.array([[1.5, 2.0], [99.0, 3.0]])


def test_missing_mask_ignores_non_float_observations():
    assert _missing_mask(np.array([[1, 2], [3, 4]])) is None


@pytest.mark.parametrize("build", [_build_loss, _build_residual])
@pytest.mark.parametrize(
    "loss, sigma, match",
    [
        ("wmse", None, "requires a sigma"),
        ("nll", None, "requires a sigma"),
        ("bogus", 1.0, "Unknown loss"),
    ],
)
def test_masked_builders_validate_arguments(build, loss, sigma, match):
    with pytest.raises(ValueError, match=match):
        build(loss, _OBS_GAP, sigma)


def test_masked_wmse_loss_and_residual_average_over_observed_entries():
    """wmse with a NaN averages over the three observed entries, and the
    residual's half-sum-of-squares equals that loss."""
    sigma = jnp.array([0.5, 1.0])
    loss = float(_build_loss("wmse", _OBS_GAP, sigma)(_PRED_GAP))
    assert loss == pytest.approx(((0.5 / 0.5) ** 2 + 0.0 + 1.0**2) / 3.0)
    r = np.asarray(_build_residual("wmse", _OBS_GAP, sigma)(_PRED_GAP))
    assert r[2] == 0.0
    assert 0.5 * np.sum(r**2) == pytest.approx(loss)


@pytest.mark.parametrize("loss", ["mse", "nll"])
def test_masked_loss_and_residual_skip_missing_entries(loss):
    """mse averages over the three observed entries and nll sums them; the
    residual is zero at the missing entry and its half-sum-of-squares equals
    the theta-dependent part of the loss."""
    sigma = jnp.array([0.5, 1.0]) if loss == "nll" else None
    value = float(_build_loss(loss, _OBS_GAP, sigma)(_PRED_GAP))
    r = np.asarray(_build_residual(loss, _OBS_GAP, sigma)(_PRED_GAP))
    assert r[2] == 0.0
    if loss == "mse":
        assert value == pytest.approx((0.5**2 + 0.0 + 1.0**2) / 3.0)
        assert 0.5 * np.sum(r**2) == pytest.approx(value)
    else:
        log_norm = 2.0 * np.log(1.0) + np.log(0.5)  # the three observed entries
        assert value == pytest.approx(log_norm + 0.5 * ((0.5 / 0.5) ** 2 + 0.0 + 1.0**2))
        assert 0.5 * np.sum(r**2) == pytest.approx(value - log_norm)
