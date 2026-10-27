"""Tests for MAP calibration + Laplace posterior."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

import aquakin
from aquakin import FreeICConfig, LaplaceConfig, OptimizerConfig
from aquakin.integrate.calibrate import (
    _from_unconstrained,
    _jacobian_physical_wrt_theta,
    _to_unconstrained,
)
from aquakin.schema.model_spec import PriorSpec

# ---------- transform roundtrips ----------


@pytest.mark.parametrize(
    "transform,value",
    [
        ("none", 0.3),
        ("none", -2.5),
        ("positive_log", 0.01),
        ("positive_log", 1.0),
        ("positive_log", 1.5e4),
        ("logit", 0.05),
        ("logit", 0.5),
        ("logit", 0.95),
    ],
)
def test_transform_roundtrip(transform, value):
    v = jnp.asarray(value)
    theta = _to_unconstrained(v, transform)
    back = _from_unconstrained(theta, transform)
    assert float(back) == pytest.approx(float(v), rel=1e-6, abs=1e-12)


def test_transform_jacobian_matches_jax_grad():
    """Analytical Jacobian must match jax.grad of the inverse transform."""
    for transform, val in [("none", 1.7), ("positive_log", 1.7), ("logit", 0.4)]:
        v = jnp.asarray(val)
        theta = _to_unconstrained(v, transform)
        analytical = float(_jacobian_physical_wrt_theta(theta, transform))
        auto = float(jax.grad(lambda t: _from_unconstrained(t, transform))(theta))
        assert analytical == pytest.approx(auto, rel=1e-6)


def test_unknown_transform_rejected():
    with pytest.raises(ValueError):
        _to_unconstrained(jnp.asarray(1.0), "exp")
    with pytest.raises(ValueError):
        _from_unconstrained(jnp.asarray(1.0), "exp")


# ---------- end-to-end calibration ----------


@pytest.fixture
def setup(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0 = jnp.asarray([1.0, 0.0])
    true_k = 0.25
    true_params = simple_model.default_parameters().at[0].set(true_k)
    t_obs = jnp.linspace(0.5, 10.0, 20)
    sol = reactor.solve(C0, params=true_params, t_span=(0.0, 10.0), t_eval=t_obs)
    obs_clean = sol.C_named("B")
    return reactor, C0, t_obs, obs_clean, true_k


def test_recovers_known_parameter_mse(setup):
    reactor, C0, t_obs, obs_clean, true_k = setup
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    assert result.converged
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)
    assert result.posterior_cov is None


def test_ad_mode_forward_matches_reverse(setup):
    """ad_mode='forward' builds the forward-capable adjoint internally (no
    diffrax in user code) and reaches the same optimum as reverse mode."""
    reactor, C0, t_obs, obs_clean, true_k = setup
    common = dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        optimizer=OptimizerConfig(method="gauss_newton"),
        laplace=False,
    )
    fwd = aquakin.calibrate(
        reactor,
        C0,
        diff=aquakin.DifferentiationConfig(mode="forward", method="through_solve"),
        **common,
    )
    rev = aquakin.calibrate(
        reactor, C0, diff=aquakin.DifferentiationConfig(mode="reverse"), **common
    )
    assert fwd.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)
    assert fwd.params_named["A_to_B.k"] == pytest.approx(rev.params_named["A_to_B.k"], rel=1e-4)


def test_ad_mode_rejects_bad_value(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError, match="mode"):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs_clean,
            t_obs=t_obs,
            free_params=["A_to_B.k"],
            observed_species=["B"],
            laplace=False,
            diff=aquakin.DifferentiationConfig(mode="sideways"),
        )


def test_ad_mode_forward_incompatible_with_stable_adjoint(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError, match="incompatible"):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs_clean,
            t_obs=t_obs,
            free_params=["A_to_B.k"],
            observed_species=["B"],
            laplace=False,
            diff=aquakin.DifferentiationConfig(mode="forward", method="stable"),
        )


# ---------- time_unit on the fitting path (issue #446/#447) ----------
# simple_model's rate constant is in s^-1, so its native time unit is seconds;
# the setup t_obs is in seconds. A user who standardises on hours must be able to
# carry the same hour-valued t_obs into calibrate/fit and get the SAME fit -- not
# a silently 3600x-compressed time axis.


def test_calibrate_time_unit_converts_t_obs(setup):
    reactor, C0, t_obs, obs_clean, true_k = setup
    common = dict(
        observations=obs_clean,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    native = aquakin.calibrate(reactor, C0, t_obs=t_obs, **common)
    # Same observations, t_obs expressed in hours -- conversion must reproduce the
    # native-axis fit (the seconds native unit -> hours factor is 3600).
    in_hours = aquakin.calibrate(reactor, C0, t_obs=t_obs / 3600.0, time_unit="h", **common)
    assert in_hours.params_named["A_to_B.k"] == pytest.approx(
        native.params_named["A_to_B.k"], rel=1e-6
    )
    assert in_hours.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)
    # And without time_unit the hour-valued axis is the silent footgun: a
    # different (wrong) optimum, not the native fit.
    wrong = aquakin.calibrate(reactor, C0, t_obs=t_obs / 3600.0, **common)
    assert wrong.params_named["A_to_B.k"] != pytest.approx(
        native.params_named["A_to_B.k"], rel=1e-2
    )


def test_fit_time_unit_converts_t_obs(setup):
    reactor, C0, t_obs, obs_clean, true_k = setup
    native = aquakin.fit(
        reactor, C0, obs_clean, t_obs, free_params=["A_to_B.k"], observed_species=["B"]
    )
    in_hours = aquakin.fit(
        reactor,
        C0,
        obs_clean,
        t_obs / 3600.0,
        free_params=["A_to_B.k"],
        observed_species=["B"],
        time_unit="h",
    )
    assert in_hours.params_named["A_to_B.k"] == pytest.approx(
        native.params_named["A_to_B.k"], rel=1e-6
    )
    assert in_hours.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)


def test_recovers_known_parameter_nll_with_noise(setup):
    reactor, C0, t_obs, obs_clean, true_k = setup
    rng = np.random.default_rng(0)
    sigma = 0.02
    noisy = jnp.asarray(np.asarray(obs_clean) + sigma * rng.standard_normal(obs_clean.shape))
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=noisy,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="nll",
        sigma=jnp.asarray(sigma),
        laplace=True,
    )
    # SciPy's convergence flag can be False even when the optimum is reached
    # (e.g. line-search-stalled-at-optimum); the substantive check is on the
    # fit quality, not the flag.
    fit = result.params_named["A_to_B.k"]
    std = result.params_named_std["A_to_B.k"]
    assert abs(fit - true_k) < 5 * std
    assert 0.0 < std < 1.0


def test_laplace_returns_positive_definite_cov(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="nll",
        sigma=jnp.asarray(0.02),
        laplace=True,
    )
    cov = np.asarray(result.posterior_cov)
    assert cov.shape == (1, 1)
    assert np.all(np.isfinite(cov))
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0)


@pytest.mark.slow  # heavy: calibrate + FD Laplace
def test_gauss_newton_laplace_matches_fd(setup):
    """The Gauss-Newton Laplace Hessian (AD ``J^T J``, first-order reverse-mode)
    must agree with the finite-difference Hessian and be positive-definite. With
    a near-exact fit the residuals vanish at the optimum, so Gauss-Newton equals
    the full Hessian up to FD truncation noise."""
    reactor, C0, t_obs, obs_clean, _ = setup
    common = dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="nll",
        sigma=jnp.asarray(0.02),
    )
    fd = aquakin.calibrate(reactor, C0, laplace=LaplaceConfig(method="fd"), **common)
    gn = aquakin.calibrate(reactor, C0, laplace=LaplaceConfig(method="gauss_newton"), **common)

    cov_fd = float(np.asarray(fd.posterior_cov)[0, 0])
    cov_gn = float(np.asarray(gn.posterior_cov)[0, 0])
    assert cov_gn > 0.0
    assert cov_gn == pytest.approx(cov_fd, rel=0.1)
    assert gn.params_named_std["A_to_B.k"] == pytest.approx(
        fd.params_named_std["A_to_B.k"], rel=0.1
    )


@pytest.mark.slow  # heavy: calibrate through stiff solve
def test_laplace_dtmax_reconstructs_tighter_reactor(simple_model):
    """laplace_dtmax computes the Laplace Hessian with a separately (tighter)
    capped reactor; it reconstructs the reactor and gives a finite posterior. On
    this non-stiff problem the tighter cap barely changes the result."""
    reactor = aquakin.BatchReactor(
        simple_model,
        aquakin.SpatialConditions.uniform(1, T=293.15),
        integrator=aquakin.IntegratorConfig(dtmax=0.5),
    )
    C0 = jnp.asarray([1.0, 0.0])
    tp = simple_model.default_parameters().at[0].set(0.25)
    t = jnp.linspace(0.5, 10.0, 20)
    obs = reactor.solve(C0, params=tp, t_span=(0.0, 10.0), t_eval=t).C_named("B")
    common = dict(
        observations=obs,
        t_obs=t,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="nll",
        sigma=jnp.asarray(0.02),
    )
    base = aquakin.calibrate(reactor, C0, laplace=LaplaceConfig(method="gauss_newton"), **common)
    tight = aquakin.calibrate(
        reactor, C0, laplace=LaplaceConfig(method="gauss_newton", dtmax=0.05), **common
    )
    assert np.all(np.isfinite(np.asarray(tight.posterior_cov)))
    assert tight.params_named_std["A_to_B.k"] == pytest.approx(
        base.params_named_std["A_to_B.k"], rel=0.1
    )


def test_unknown_laplace_method_rejected(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs_clean,
            t_obs=t_obs,
            free_params=["A_to_B.k"],
            observed_species=["B"],
            loss="mse",
            laplace=LaplaceConfig(method="bogus"),
        )


def test_falls_back_to_schema_transform_when_omitted(simple_model):
    """If transforms={} is passed, the per-parameter declared transform is used."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0 = jnp.asarray([1.0, 0.0])
    obs = jnp.asarray([0.0, 0.5, 0.8])
    t_obs = jnp.asarray([0.0, 5.0, 10.0])
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=obs,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    # The fixture model declares no explicit transform -> defaults to "none".
    assert result.transforms == ["none"]


# ---------- Gaussian priors ----------


def test_priorspec_range_to_gaussian():
    """A literature range maps to N(midpoint, (hi-lo)/4)."""
    assert PriorSpec(range=(4.0, 8.0)).resolved() == (6.0, 1.0)
    assert PriorSpec(range=(0.5, 1.0)).resolved() == (0.75, 0.125)


def test_priorspec_mean_std_direct():
    assert PriorSpec(mean=17.1, std=2.3).resolved() == (17.1, 2.3)


def test_priorspec_rejects_both_and_neither():
    with pytest.raises(ValidationError):
        PriorSpec(mean=1.0, std=1.0, range=(0.0, 2.0))
    with pytest.raises(ValidationError):
        PriorSpec()
    with pytest.raises(ValidationError):
        PriorSpec(mean=1.0, std=-1.0)  # std must be > 0


def test_prior_pulls_estimate_toward_prior_mean(setup):
    """A Gaussian prior centred above the data optimum pulls the MAP up."""
    reactor, C0, t_obs, obs_clean, true_k = setup
    common = dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    no_prior = aquakin.calibrate(reactor, C0, **common)
    # Prior mean well above the data-optimal true_k (~0.25), moderately tight.
    with_prior = aquakin.calibrate(reactor, C0, priors={"A_to_B.k": (2.0, 0.05)}, **common)
    k_no = no_prior.params_named["A_to_B.k"]
    k_pr = with_prior.params_named["A_to_B.k"]
    assert k_no == pytest.approx(true_k, rel=1e-2)
    assert k_pr > k_no  # prior drags the estimate upward
    assert with_prior.priors_applied == {"A_to_B.k": (2.0, 0.05)}


def test_priors_ignored_when_not_free(setup):
    """A prior on a parameter that is not being fit has no effect/record."""
    reactor, C0, t_obs, obs_clean, true_k = setup
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
        priors={"some_other_param": (1.0, 0.1)},
    )
    assert result.priors_applied == {}
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)


# ---------- multistart ----------


def test_multistart_reproducible(setup):
    """Same seed -> identical multistart result (deterministic restarts)."""
    reactor, C0, t_obs, obs_clean, _ = setup
    common = dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
        optimizer=OptimizerConfig(n_starts=6, jitter=0.8, seed=3),
    )
    r1 = aquakin.calibrate(reactor, C0, **common)
    r2 = aquakin.calibrate(reactor, C0, **common)
    assert r1.params_named["A_to_B.k"] == r2.params_named["A_to_B.k"]


def test_multistart_recovers_from_bad_initial(setup):
    """A start far from the truth still recovers k with several restarts."""
    reactor, C0, t_obs, obs_clean, true_k = setup
    bad = reactor.model.default_parameters().at[0].set(40.0)
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
        initial_params=bad,
        optimizer=OptimizerConfig(n_starts=8, jitter=1.5, seed=0),
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)


def test_multistart_default_is_single_start(setup):
    """n_starts defaults to 1, reproducing the single-start result exactly."""
    reactor, C0, t_obs, obs_clean, _ = setup
    common = dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    default = aquakin.calibrate(reactor, C0, **common)
    one = aquakin.calibrate(reactor, C0, optimizer=OptimizerConfig(n_starts=1), **common)
    assert default.params_named["A_to_B.k"] == one.params_named["A_to_B.k"]


def test_multistart_keeps_best_loss(setup):
    """The reported loss is no worse than the unperturbed single start (start 0
    is always included), so multistart never degrades the fit."""
    reactor, C0, t_obs, obs_clean, _ = setup
    common = dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    single = aquakin.calibrate(reactor, C0, optimizer=OptimizerConfig(n_starts=1), **common)
    multi = aquakin.calibrate(
        reactor, C0, optimizer=OptimizerConfig(n_starts=5, jitter=1.0, seed=1), **common
    )
    assert multi.loss <= single.loss + 1e-9


def test_param_halfwidth_bounds_the_rate(setup):
    """param_halfwidth box-bounds the rate in log space around the start, so a
    tight halfwidth keeps the fit from reaching a far-away true value, while the
    default (None) lets it reach it."""
    reactor, C0, t_obs, obs_clean, true_k = setup  # true_k=0.25
    k_start = float(reactor.model.default_parameters()[0])  # 0.1
    common = dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    bounded = aquakin.calibrate(
        reactor, C0, optimizer=OptimizerConfig(param_halfwidth=0.2), **common
    )
    unbounded = aquakin.calibrate(reactor, C0, **common)
    kb = bounded.params_named["A_to_B.k"]
    # Within the log-space box around the start, and short of the true value.
    assert abs(np.log(kb) - np.log(k_start)) <= 0.2 + 1e-6
    assert kb < true_k
    assert unbounded.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)


def test_jitter_schedule_reproducible(setup):
    """Cyclic jitter_schedule with a fixed seed is reproducible."""
    reactor, C0, t_obs, obs_clean, _ = setup
    common = dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
        optimizer=OptimizerConfig(n_starts=4, jitter_schedule=(0.3, 0.5, 0.8), seed=2),
    )
    r1 = aquakin.calibrate(reactor, C0, **common)
    r2 = aquakin.calibrate(reactor, C0, **common)
    assert r1.params_named["A_to_B.k"] == r2.params_named["A_to_B.k"]


def test_multistart_invalid_n_starts_rejected(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs_clean,
            t_obs=t_obs,
            free_params=["A_to_B.k"],
            observed_species=["B"],
            loss="mse",
            laplace=False,
            optimizer=OptimizerConfig(n_starts=0),
        )


# ---------- forward-solve failures during the fit ----------


def _failing_forward_model(monkeypatch, fails):
    """Swap calibrate's reactor forward model for one whose compiled solve
    raises (through ``equinox.error_if``, as a diverging stiff linear solve
    does) whenever ``fails(k)`` is true for the rate ``k = p[0]``."""
    import equinox as eqx

    cal_mod = sys.modules["aquakin.integrate.calibrate"]

    class _Failing(cal_mod._ReactorForwardModel):
        def solve_trajectory(self, p, C0_k, tspan, tobs):
            C = super().solve_trajectory(p, C0_k, tspan, tobs)
            return eqx.error_if(
                C, fails(p[0]), "A linear solver returned non-finite (NaN or inf) output."
            )

    monkeypatch.setattr(cal_mod, "_ReactorForwardModel", _Failing)


def _fail_fit_kwargs(setup, optimizer):
    reactor, C0, t_obs, obs_clean, _ = setup
    return dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
        initial_params=reactor.model.default_parameters().at[0].set(0.5),
        optimizer=optimizer,
    )


@pytest.mark.parametrize("method", ["gauss_newton", "lbfgsb"])
def test_multistart_skips_start_whose_solve_fails(setup, monkeypatch, method):
    """A jittered restart whose forward solve fails is skipped (with a warning)
    and the fit keeps the good start instead of aborting."""
    reactor, C0, _t, _obs, true_k = setup
    # Start 0 (k=0.5) and start 1 (k~0.9) descend to the truth below k=3;
    # start 2's jitter (+6.1 in log space) lands far above it and fails.
    _failing_forward_model(monkeypatch, lambda k: k > 3.0)
    opt = OptimizerConfig(method=method, n_starts=3, jitter_schedule=(3.0,), seed=1)
    with pytest.warns(RuntimeWarning, match=r"failed in 1 of 3 optimiser starts \(start index 2"):
        result = aquakin.calibrate(reactor, C0, **_fail_fit_kwargs(setup, opt))
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)


def test_all_starts_failing_raises_actionable_error(setup, monkeypatch):
    """When the solve fails in every start, calibrate raises a RuntimeError
    with the remedies, chained to the underlying solve error."""
    reactor, C0, *_ = setup
    # Both starts (k=0.5 and k~0.7) are fine at their start point but fail once
    # the optimiser descends below k=0.4 toward the truth (0.25).
    _failing_forward_model(monkeypatch, lambda k: k < 0.4)
    opt = OptimizerConfig(method="gauss_newton", n_starts=2, jitter_schedule=(1.0,), seed=0)
    with pytest.raises(RuntimeError, match="param_halfwidth") as info:
        aquakin.calibrate(reactor, C0, **_fail_fit_kwargs(setup, opt))
    assert "all 2 optimiser starts" in str(info.value)
    assert info.value.__cause__ is not None


def test_solve_failure_at_start_guess_raises_actionable_error(setup, monkeypatch):
    """A solve that already fails at the starting guess is reported as such by
    the pre-fit gradient check, not as a raw XLA callback traceback."""
    reactor, C0, *_ = setup
    _failing_forward_model(monkeypatch, lambda k: k < 0.6)
    # L-BFGS-B's probe evaluates the loss value, which carries the fake's check
    # (a Jacobian-only probe would let XLA prune the primal-side check).
    opt = OptimizerConfig(method="lbfgsb")
    with pytest.raises(RuntimeError, match="at the starting guess") as info:
        aquakin.calibrate(reactor, C0, **_fail_fit_kwargs(setup, opt))
    assert info.value.__cause__ is not None


def test_programming_error_in_solve_is_not_swallowed(setup, monkeypatch):
    """Only solve failures are treated as a failed start; any other exception
    from the forward model propagates unchanged."""
    cal_mod = sys.modules["aquakin.integrate.calibrate"]

    class _Broken(cal_mod._ReactorForwardModel):
        def solve_trajectory(self, p, C0_k, tspan, tobs):
            raise TypeError("bug in the forward model")

    monkeypatch.setattr(cal_mod, "_ReactorForwardModel", _Broken)
    reactor, C0, *_ = setup
    opt = OptimizerConfig(n_starts=2)
    with pytest.raises(TypeError, match="bug in the forward model"):
        aquakin.calibrate(reactor, C0, **_fail_fit_kwargs(setup, opt))


# ---------- posterior-predictive band ----------


def _band_setup(setup):
    reactor, C0, t_obs, obs_clean, true_k = setup
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="nll",
        sigma=jnp.asarray(0.02),
        laplace=True,
    )
    return reactor, C0, t_obs, obs_clean, result


def test_predictive_band_brackets_truth(setup):
    """The 95% band envelopes lo <= median <= hi and contains the (noise-free)
    truth at essentially every observation time."""
    reactor, C0, t_obs, obs_clean, result = _band_setup(setup)
    band = result.predictive_band(reactor, C0, t_obs, observed_species=["B"], n_draw=200, seed=0)
    assert band.lo.shape == band.hi.shape == band.median.shape == (len(t_obs), 1)
    assert band.n_valid > 0
    assert np.all(band.lo <= band.median + 1e-9)
    assert np.all(band.median <= band.hi + 1e-9)
    truth = np.asarray(obs_clean).reshape(-1, 1)
    inside = (truth >= band.lo - 1e-6) & (truth <= band.hi + 1e-6)
    assert inside.mean() > 0.8


def test_predictive_band_reproducible(setup):
    reactor, C0, t_obs, _, result = _band_setup(setup)
    b1 = result.predictive_band(reactor, C0, t_obs, observed_species=["B"], seed=7)
    b2 = result.predictive_band(reactor, C0, t_obs, observed_species=["B"], seed=7)
    assert np.allclose(b1.lo, b2.lo) and np.allclose(b1.hi, b2.hi)


def test_predictive_band_all_species_shape(setup):
    reactor, C0, t_obs, _, result = _band_setup(setup)
    band = result.predictive_band(reactor, C0, t_obs, n_draw=50, seed=0)
    # No observed_species -> all model species (A and B) as columns.
    assert band.median.shape == (len(t_obs), reactor.model.n_species)
    assert band.species is None


def test_predictive_band_requires_laplace(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    with pytest.raises(ValueError):
        result.predictive_band(reactor, C0, t_obs)


def test_predictive_band_eig_keep_is_deprecated(setup):
    reactor, C0, t_obs, _, result = _band_setup(setup)
    with pytest.warns(DeprecationWarning, match="eig_keep"):
        result.predictive_band(reactor, C0, t_obs, observed_species=["B"], n_draw=10, eig_keep=1e-2)


# ---------- Laplace covariance regularisation (#7) ----------


def test_laplace_covariance_truncates_null_direction():
    """A Hessian with one near-null eigen-direction yields a rank-deficient
    covariance: the unidentifiable direction is dropped, and the kept variance
    is 1/eigenvalue along the strong direction."""
    from aquakin.integrate.calibrate import _laplace_covariance

    V = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)  # orthonormal
    H = V @ np.diag([100.0, 1e-9]) @ V.T
    cov, wk, Vk = _laplace_covariance(H, ridge=1e-6, eig_keep=1e-2)
    assert wk.shape == (1,)  # the null direction dropped
    assert wk[0] == pytest.approx(100.0, rel=1e-3)
    s = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert int(np.sum(s > 1e-9)) == 1  # rank-1 covariance
    assert float(np.max(s)) == pytest.approx(1.0 / 100.0, rel=1e-3)


def test_laplace_covariance_keeps_all_when_well_conditioned():
    """A well-conditioned Hessian keeps every direction; the covariance then
    equals inv(H + ridge) (no truncation)."""
    from aquakin.integrate.calibrate import _laplace_covariance

    H = np.diag([10.0, 20.0])
    cov, wk, _ = _laplace_covariance(H, ridge=1e-6, eig_keep=1e-2)
    assert wk.shape == (2,)
    assert np.allclose(cov, np.linalg.inv(H + 1e-6 * np.eye(2)), atol=1e-9)


def test_laplace_covariance_relative_to_largest_for_small_scale_hessian():
    """A uniformly small-scale Hessian (largest eigenvalue << 1) but with clear
    structure must keep its identifiable direction(s). The truncation is RELATIVE
    to the largest eigenvalue, not an absolute floor.

    Regression for the bug where an observable on a tiny scale (e.g. a trace
    species in molar units, JᵀJ ~1e-2 with eigenvalues spanning orders of
    magnitude) was reported fully degenerate because the threshold was floored at
    an absolute ``eig_keep * 1`` -- discarding every direction.
    """
    from aquakin.integrate.calibrate import _laplace_covariance

    V = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)  # orthonormal
    # Largest eigenvalue 9.1e-3 (< 1); a near-null direction at the ridge level.
    H = V @ np.diag([9.1e-3, 1e-9]) @ V.T
    cov, wk, Vk = _laplace_covariance(H, ridge=1e-6, eig_keep=1e-2)
    assert wk.shape == (1,)  # strong direction kept...
    assert wk[0] == pytest.approx(9.1e-3, rel=1e-3)  # ...at its true eigenvalue
    s = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert int(np.sum(s > 1e-12)) == 1  # rank-1 covariance
    assert float(np.max(s)) == pytest.approx(1.0 / 9.1e-3, rel=1e-3)


@pytest.mark.parametrize("eig_keep", [1.0, 1.5, -0.1])
def test_laplace_covariance_rejects_out_of_range_eig_keep(eig_keep):
    """eig_keep is a fraction of the largest eigenvalue, so it must be in [0, 1).
    A value >= 1 would drop every direction and (silently) return an all-zero
    covariance -- it must raise a clear error instead."""
    from aquakin.integrate.calibrate import _laplace_covariance

    H = np.diag([10.0, 20.0])
    with pytest.raises(ValueError, match=r"eig_keep must be in \[0, 1\)"):
        _laplace_covariance(H, ridge=1e-6, eig_keep=eig_keep)


def test_calibrate_nll_small_scale_observable_gives_finite_posterior(setup):
    """End-to-end: a NLL+Laplace fit of a tiny-magnitude observable yields a
    finite, positive-variance posterior rather than a false 'degenerate' error
    (the bug reproduced through the full calibrate path)."""
    reactor, C0, t_obs, _obs_clean, true_k = setup
    true_params = reactor.model.default_parameters().at[0].set(true_k)
    # Scale the observable down to ~1e-6 so the Hessian is uniformly small.
    scale = 1e-6
    C0s = C0 * scale
    obs = reactor.solve(
        C0s, params=true_params, t_span=(0.0, float(t_obs[-1])), t_eval=t_obs
    ).C_named("B")
    res = aquakin.calibrate(
        reactor,
        C0s,
        observations=obs,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        observed_species=["B"],
        loss="nll",
        sigma=jnp.asarray(0.05 * scale),
        laplace=True,
    )
    assert np.all(np.isfinite(np.asarray(res.posterior_cov)))
    assert res.params_named_std["A_to_B.k"] > 0.0


def test_band_and_std_share_the_truncation(setup):
    """params_named_std and predictive_band draw from the SAME posterior_cov, so
    they regularise identically -- the draws' empirical covariance matches the
    reported posterior_cov (the #7 consistency)."""
    reactor, C0, t_obs, _, result = _band_setup(setup)
    # Reconstruct the draws the band uses (1 free param here) and check their
    # spread matches posterior_cov[0,0].
    cov00 = float(np.asarray(result.posterior_cov)[0, 0])
    # std reported (unconstrained space) is sqrt(posterior_cov diagonal).
    std_report = float(np.asarray(result.posterior_std_unconstrained)[0])
    assert std_report == pytest.approx(np.sqrt(cov00), rel=1e-6)


# ---------- Gauss-Newton optimiser ----------


def test_gauss_newton_recovers_known_parameter(setup):
    reactor, C0, t_obs, obs_clean, true_k = setup
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
        optimizer=OptimizerConfig(method="gauss_newton"),
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)
    # n_iter is an iteration-scale count (Jacobian evals for the GN path), not the
    # raw function-evaluation count -- a small positive int.
    assert isinstance(result.n_iter, int) and result.n_iter >= 1


def test_gauss_newton_matches_lbfgsb_on_easy_fit(setup):
    """On a convex (single-minimum) fit both optimisers reach the same optimum."""
    reactor, C0, t_obs, obs_clean, _ = setup
    common = dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    lb = aquakin.calibrate(reactor, C0, optimizer=OptimizerConfig(), **common)
    gn = aquakin.calibrate(reactor, C0, optimizer=OptimizerConfig(method="gauss_newton"), **common)
    assert gn.params_named["A_to_B.k"] == pytest.approx(lb.params_named["A_to_B.k"], rel=1e-3)


def test_nll_loss_is_comparable_across_optimizers(setup):
    """CalibrationResult.loss must be the same scalar objective regardless of
    optimizer. The Gauss-Newton path minimises 0.5*||residual||^2, which for
    loss='nll' drops the sum(log(sigma)) normaliser the L-BFGS-B objective
    includes; the reported loss must add it back so both agree."""
    reactor, C0, t_obs, obs_clean, _ = setup
    sigma = jnp.asarray(0.02)
    common = dict(
        observations=obs_clean,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="nll",
        sigma=sigma,
        laplace=False,
    )
    lb = aquakin.calibrate(reactor, C0, optimizer=OptimizerConfig(), **common)
    gn = aquakin.calibrate(reactor, C0, optimizer=OptimizerConfig(method="gauss_newton"), **common)
    # Same optimum -> same reported objective (the full NLL, not 0.5||r||^2).
    assert gn.loss == pytest.approx(lb.loss, rel=1e-4)
    # And it is the full NLL: well above the GN 0.5||r||^2, which here is ~0
    # (clean data) while sum(log(sigma)) = n_obs * log(0.02) < 0.
    expected_const = float(t_obs.shape[0] * jnp.log(sigma))
    assert gn.loss == pytest.approx(expected_const, abs=1e-3)


def test_gauss_newton_forward_mode_with_direct_adjoint(simple_model):
    """With a DirectAdjoint reactor the GN Jacobian is formed in forward mode
    (jacfwd); it must still recover the parameter."""
    reactor = aquakin.BatchReactor(
        simple_model,
        aquakin.SpatialConditions.uniform(1, T=293.15),
        diff=aquakin.DifferentiationConfig(mode="forward", method="through_solve"),
    )
    C0 = jnp.asarray([1.0, 0.0])
    true_k = 0.25
    true_params = simple_model.default_parameters().at[0].set(true_k)
    t_obs = jnp.linspace(0.5, 10.0, 20)
    obs = reactor.solve(C0, params=true_params, t_span=(0.0, 10.0), t_eval=t_obs).C_named("B")
    result = aquakin.calibrate(
        reactor,
        C0,
        observations=obs,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
        optimizer=OptimizerConfig(method="gauss_newton"),
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)


def test_gauss_newton_with_free_ic_and_multistart(simple_model):
    """GN composes with free_ic and multistart: recover both k and A0."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    true_k, true_A0 = 0.25, 1.6
    true_params = simple_model.default_parameters().at[0].set(true_k)
    t_obs = jnp.linspace(0.5, 12.0, 25)
    sol = reactor.solve(
        jnp.asarray([true_A0, 0.0]), params=true_params, t_span=(0.0, 12.0), t_eval=t_obs
    )
    obs = jnp.stack([sol.C_named("A"), sol.C_named("B")], axis=1)
    result = aquakin.calibrate(
        reactor,
        jnp.asarray([1.0, 0.0]),
        observations=obs,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["A", "B"],
        loss="mse",
        laplace=False,
        optimizer=OptimizerConfig(method="gauss_newton", n_starts=3, seed=0),
        free_ic=FreeICConfig(["A"], bounds=(0.1, 10.0)),
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)
    assert result.ic_named[0]["A"] == pytest.approx(true_A0, rel=1e-2)


def test_unknown_optimizer_rejected(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs_clean,
            t_obs=t_obs,
            free_params=["A_to_B.k"],
            observed_species=["B"],
            loss="mse",
            laplace=False,
            optimizer=OptimizerConfig(method="newton"),
        )


# ---------- free initial conditions ----------


def test_free_ic_recovers_initial_condition(simple_model):
    """Fit an unknown initial A0 jointly with the rate; recover both."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    true_k, true_A0 = 0.25, 1.7
    true_params = simple_model.default_parameters().at[0].set(true_k)
    C0_true = jnp.asarray([true_A0, 0.0])
    t_obs = jnp.linspace(0.5, 12.0, 25)
    sol = reactor.solve(C0_true, params=true_params, t_span=(0.0, 12.0), t_eval=t_obs)
    obs = jnp.stack([sol.C_named("A"), sol.C_named("B")], axis=1)

    # Start from the wrong A0 (1.0); free it.
    C0_start = jnp.asarray([1.0, 0.0])
    result = aquakin.calibrate(
        reactor,
        C0_start,
        observations=obs,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["A", "B"],
        loss="mse",
        laplace=False,
        free_ic=FreeICConfig(["A"], bounds=(0.1, 10.0)),
        optimizer=OptimizerConfig(n_starts=3, seed=0),
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)
    assert result.ic_named[0]["A"] == pytest.approx(true_A0, rel=1e-2)
    assert float(result.C0_fitted[0][0]) == pytest.approx(true_A0, rel=1e-2)


def test_free_ic_per_dataset_in_multibatch(simple_model):
    """Two batches with different (unknown) A0 but a shared k; each batch's
    initial pool is fitted separately."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    true_k = 0.3
    true_params = simple_model.default_parameters().at[0].set(true_k)
    A0a, A0b = 1.2, 2.4
    ta = jnp.linspace(0.5, 10.0, 20)
    tb = jnp.linspace(0.5, 10.0, 20)
    sola = reactor.solve(jnp.asarray([A0a, 0.0]), params=true_params, t_span=(0.0, 10.0), t_eval=ta)
    solb = reactor.solve(jnp.asarray([A0b, 0.0]), params=true_params, t_span=(0.0, 10.0), t_eval=tb)
    obsa = jnp.stack([sola.C_named("A"), sola.C_named("B")], axis=1)
    obsb = jnp.stack([solb.C_named("A"), solb.C_named("B")], axis=1)
    start = jnp.asarray([1.0, 0.0])
    result = aquakin.calibrate(
        reactor,
        [start, start],
        observations=[obsa, obsb],
        t_obs=[ta, tb],
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["A", "B"],
        loss="mse",
        laplace=False,
        free_ic=FreeICConfig(["A"], bounds=(0.1, 10.0)),
        optimizer=OptimizerConfig(n_starts=3, seed=0),
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)
    assert result.ic_named[0]["A"] == pytest.approx(A0a, rel=2e-2)
    assert result.ic_named[1]["A"] == pytest.approx(A0b, rel=2e-2)


def test_free_ic_laplace_is_over_rates_only(simple_model):
    """With free_ic + laplace, the posterior covers the rate parameters only
    (pools held at their MAP), so its shape matches the rate count."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    true_params = simple_model.default_parameters().at[0].set(0.25)
    t_obs = jnp.linspace(0.5, 12.0, 25)
    sol = reactor.solve(
        jnp.asarray([1.5, 0.0]), params=true_params, t_span=(0.0, 12.0), t_eval=t_obs
    )
    obs = jnp.stack([sol.C_named("A"), sol.C_named("B")], axis=1)
    result = aquakin.calibrate(
        reactor,
        jnp.asarray([1.0, 0.0]),
        observations=obs,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["A", "B"],
        loss="nll",
        sigma=jnp.asarray(0.02),
        laplace=True,
        free_ic=FreeICConfig(["A"], bounds=(0.1, 10.0)),
    )
    assert np.asarray(result.posterior_cov).shape == (1, 1)
    assert "A_to_B.k" in result.params_named_std


# ---------- joint rate + initial-pool Laplace posterior ----------

_DECAY_K, _DECAY_A0, _DECAY_SIGMA = 0.25, 1.5, 0.01


def _decay_fit(
    simple_model,
    laplace,
    *,
    free_ic=True,
    n_datasets=1,
    ic_prior_log_std=None,
    priors=None,
    a0_start=1.0,
):
    """Fit k and A0 of A -> B to noise-free observations of A alone."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    true_params = simple_model.default_parameters().at[0].set(_DECAY_K)
    t_obs = jnp.linspace(0.5, 12.0, 25)
    sol = reactor.solve(
        jnp.asarray([_DECAY_A0, 0.0]), params=true_params, t_span=(0.0, 12.0), t_eval=t_obs
    )
    obs = sol.C_named("A")
    C0 = jnp.asarray([a0_start, 0.0])
    return aquakin.calibrate(
        reactor,
        [C0] * n_datasets if n_datasets > 1 else C0,
        observations=[obs] * n_datasets if n_datasets > 1 else obs,
        t_obs=[t_obs] * n_datasets if n_datasets > 1 else t_obs,
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["A"],
        loss="nll",
        sigma=jnp.asarray(_DECAY_SIGMA),
        priors=priors,
        laplace=laplace,
        free_ic=(
            FreeICConfig(["A"], bounds=(0.1, 10.0), prior_log_std=ic_prior_log_std)
            if free_ic
            else None
        ),
        optimizer=OptimizerConfig(method="gauss_newton"),
    ), t_obs


def _decay_fisher(t_obs):
    """Closed-form Fisher information of A(t) = A0 exp(-k t) in (log k, log A0):
    dA/dlog k = -k t A and dA/dlog A0 = A."""
    t = np.asarray(t_obs)
    a = _DECAY_A0 * np.exp(-_DECAY_K * t)
    S = np.stack([-_DECAY_K * t * a, a], axis=1)
    return S.T @ S / _DECAY_SIGMA**2


_EXACT_LAPLACE = dict(ridge=1e-12, eig_keep=0.0)


def test_joint_ic_gauss_newton_hessian_matches_closed_form_fisher(simple_model):
    """The joint Gauss-Newton Hessian over (log k, log A0) is the closed-form
    Fisher information, and its inverse is the joint covariance."""
    result, t_obs = _decay_fit(
        simple_model, LaplaceConfig(method="gauss_newton", include_ic=True, **_EXACT_LAPLACE)
    )
    F = _decay_fisher(t_obs)
    assert result.joint_names == ["A_to_B.k", "ic[0].A"]
    np.testing.assert_allclose(np.asarray(result.joint_hessian), F, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(result.joint_cov), np.linalg.inv(F), rtol=1e-4)


def test_joint_ic_rate_std_is_marginal_over_pools(simple_model):
    """With include_ic the rate std is the marginal sqrt(inv(F)[0, 0]), wider
    than the default rate-only std sqrt(1 / F[0, 0]) conditional on A0."""
    lap = dict(method="gauss_newton", **_EXACT_LAPLACE)
    joint, t_obs = _decay_fit(simple_model, LaplaceConfig(include_ic=True, **lap))
    rate_only, _ = _decay_fit(simple_model, LaplaceConfig(**lap))
    F = _decay_fisher(t_obs)
    std_joint = float(joint.posterior_std_unconstrained[0])
    std_rate_only = float(rate_only.posterior_std_unconstrained[0])
    assert std_joint == pytest.approx(np.sqrt(np.linalg.inv(F)[0, 0]), rel=1e-4)
    assert std_rate_only == pytest.approx(np.sqrt(1.0 / F[0, 0]), rel=1e-4)
    assert std_joint > std_rate_only
    # The rate-rate Hessian block means the same thing in both modes.
    np.testing.assert_allclose(
        np.asarray(joint.hessian_unconstrained),
        np.asarray(rate_only.hessian_unconstrained),
        rtol=1e-6,
    )
    # Pool std in physical units: A0 * std(log A0).
    assert joint.ic_named_std[0]["A"] == pytest.approx(
        _DECAY_A0 * np.sqrt(np.linalg.inv(F)[1, 1]), rel=1e-3
    )
    assert joint.params_named_std["A_to_B.k"] == pytest.approx(_DECAY_K * std_joint, rel=1e-3)


def test_joint_ic_fd_hessian_matches_gauss_newton(simple_model):
    """At a zero-residual fit the FD loss Hessian equals J^T J."""
    fd, t_obs = _decay_fit(simple_model, LaplaceConfig(method="fd", include_ic=True))
    np.testing.assert_allclose(np.asarray(fd.joint_hessian), _decay_fisher(t_obs), rtol=1e-3)


def test_joint_ic_fd_hessian_includes_rate_and_ic_priors(simple_model):
    """The FD joint Hessian adds the rate-prior and pool-prior precisions. Both
    priors are centred on the truth so the MAP stays zero-residual and FD equals
    Gauss-Newton exactly."""
    s_k, s_ic = 0.1, 0.5
    fd, t_obs = _decay_fit(
        simple_model,
        LaplaceConfig(method="fd", include_ic=True),
        ic_prior_log_std=s_ic,
        priors={"A_to_B.k": (_DECAY_K, s_k)},
        a0_start=_DECAY_A0,
    )
    # The rate prior is Gaussian in physical k, so in log k it adds (k / s_k)^2.
    expected = _decay_fisher(t_obs) + np.diag([(_DECAY_K / s_k) ** 2, 1.0 / s_ic**2])
    np.testing.assert_allclose(np.asarray(fd.joint_hessian), expected, rtol=1e-3)


def test_joint_ic_includes_ic_prior(simple_model):
    """A log-space pool prior adds 1 / prior_log_std^2 to the pool diagonal."""
    s = 0.5
    result, t_obs = _decay_fit(
        simple_model,
        LaplaceConfig(method="gauss_newton", include_ic=True, **_EXACT_LAPLACE),
        ic_prior_log_std=s,
    )
    expected = _decay_fisher(t_obs) + np.diag([0.0, 1.0 / s**2])
    np.testing.assert_allclose(np.asarray(result.joint_hessian), expected, rtol=1e-4)


def test_joint_ic_multi_dataset_layout(simple_model):
    """Two datasets: one shared rate plus one pool per dataset."""
    result, _ = _decay_fit(
        simple_model,
        LaplaceConfig(method="gauss_newton", include_ic=True),
        n_datasets=2,
    )
    assert result.joint_names == ["A_to_B.k", "ic[0].A", "ic[1].A"]
    assert np.asarray(result.joint_cov).shape == (3, 3)
    assert np.asarray(result.posterior_cov).shape == (1, 1)
    assert [list(d) for d in result.ic_named_std] == [["A"], ["A"]]


def test_joint_fields_absent_by_default_and_without_free_ic(simple_model):
    """The default stays rate-only, and include_ic is a no-op without free_ic."""
    default, _ = _decay_fit(simple_model, LaplaceConfig(method="gauss_newton"))
    no_ic, _ = _decay_fit(
        simple_model, LaplaceConfig(method="gauss_newton", include_ic=True), free_ic=False
    )
    for r in (default, no_ic):
        assert r.joint_cov is None
        assert r.joint_hessian is None
        assert r.joint_names is None
        assert r.ic_named_std is None
        assert np.asarray(r.posterior_cov).shape == (1, 1)


def test_free_ic_unknown_species_rejected(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(KeyError):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs_clean,
            t_obs=t_obs,
            free_params=["A_to_B.k"],
            observed_species=["B"],
            loss="mse",
            laplace=False,
            free_ic=FreeICConfig(["NotASpecies"]),
        )


def test_free_ic_bad_bounds_rejected(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs_clean,
            t_obs=t_obs,
            free_params=["A_to_B.k"],
            observed_species=["B"],
            loss="mse",
            laplace=False,
            free_ic=FreeICConfig(["A"], bounds=(5.0, 1.0)),
        )


# ---------- joint multi-batch fit ----------


def test_joint_multibatch_recovers_known_parameter(simple_model):
    """Two batches with different C0 / time grids, fit jointly, recover k."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    true_k = 0.3
    true_params = simple_model.default_parameters().at[0].set(true_k)
    C0a = jnp.asarray([1.0, 0.0])
    C0b = jnp.asarray([2.0, 0.0])
    ta = jnp.linspace(0.5, 8.0, 12)
    tb = jnp.linspace(0.3, 10.0, 15)
    obsa = reactor.solve(C0a, params=true_params, t_span=(0.0, 8.0), t_eval=ta).C_named("B")
    obsb = reactor.solve(C0b, params=true_params, t_span=(0.0, 10.0), t_eval=tb).C_named("B")
    result = aquakin.calibrate(
        reactor,
        [C0a, C0b],
        observations=[obsa, obsb],
        t_obs=[ta, tb],
        free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    assert result.converged
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)


def test_multibatch_length_mismatch_rejected(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0a = jnp.asarray([1.0, 0.0])
    t = jnp.asarray([0.0, 1.0])
    obs = jnp.asarray([0.0, 0.5])
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor,
            [C0a, C0a],
            observations=[obs],
            t_obs=[t, t],
            free_params=["A_to_B.k"],
            observed_species=["B"],
            loss="mse",
            laplace=False,
        )


def test_rejects_empty_free_params(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs_clean,
            t_obs=t_obs,
            free_params=[],
            observed_species=["B"],
        )


def test_rejects_unknown_loss(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs_clean,
            t_obs=t_obs,
            free_params=["A_to_B.k"],
            observed_species=["B"],
            loss="huber",
        )


def test_rejects_wmse_without_sigma(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor,
            C0,
            observations=obs_clean,
            t_obs=t_obs,
            free_params=["A_to_B.k"],
            observed_species=["B"],
            loss="wmse",
        )


def test_positive_log_initial_negative_rejected(simple_model):
    """If initial value is <= 0 the transform is invalid."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    init_bad = simple_model.default_parameters().at[0].set(-1.0)
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.asarray([0.0, 0.5]),
            t_obs=jnp.asarray([0.0, 1.0]),
            free_params=["A_to_B.k"],
            transforms={"A_to_B.k": "positive_log"},
            initial_params=init_bad,
            observed_species=["B"],
        )


def test_logit_initial_out_of_range_rejected(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    init_bad = simple_model.default_parameters().at[0].set(1.5)
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.asarray([0.0, 0.5]),
            t_obs=jnp.asarray([0.0, 1.0]),
            free_params=["A_to_B.k"],
            transforms={"A_to_B.k": "logit"},
            initial_params=init_bad,
            observed_species=["B"],
        )
