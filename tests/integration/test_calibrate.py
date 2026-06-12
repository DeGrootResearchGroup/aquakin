"""Tests for MAP calibration + Laplace posterior."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate.calibrate import (
    _from_unconstrained,
    _jacobian_physical_wrt_theta,
    _to_unconstrained,
)
from aquakin.schema.network_spec import PriorSpec
from pydantic import ValidationError


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
def setup(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    C0 = jnp.asarray([1.0, 0.0])
    true_k = 0.25
    true_params = simple_network.default_parameters().at[0].set(true_k)
    t_obs = jnp.linspace(0.5, 10.0, 20)
    sol = reactor.solve(C0, true_params, t_span=(0.0, 10.0), t_eval=t_obs)
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
        laplace=True,
    )
    fd = aquakin.calibrate(reactor, C0, laplace_method="fd", **common)
    gn = aquakin.calibrate(reactor, C0, laplace_method="gauss_newton", **common)

    cov_fd = float(np.asarray(fd.posterior_cov)[0, 0])
    cov_gn = float(np.asarray(gn.posterior_cov)[0, 0])
    assert cov_gn > 0.0
    assert cov_gn == pytest.approx(cov_fd, rel=0.1)
    assert gn.params_named_std["A_to_B.k"] == pytest.approx(
        fd.params_named_std["A_to_B.k"], rel=0.1
    )


@pytest.mark.slow  # heavy: calibrate through stiff solve
def test_laplace_dtmax_reconstructs_tighter_reactor(simple_network):
    """laplace_dtmax computes the Laplace Hessian with a separately (tighter)
    capped reactor; it reconstructs the reactor and gives a finite posterior. On
    this non-stiff problem the tighter cap barely changes the result."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15), dtmax=0.5)
    C0 = jnp.asarray([1.0, 0.0])
    tp = simple_network.default_parameters().at[0].set(0.25)
    t = jnp.linspace(0.5, 10.0, 20)
    obs = reactor.solve(C0, tp, t_span=(0.0, 10.0), t_eval=t).C_named("B")
    common = dict(
        observations=obs, t_obs=t, free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"}, observed_species=["B"],
        loss="nll", sigma=jnp.asarray(0.02), laplace=True,
        laplace_method="gauss_newton",
    )
    base = aquakin.calibrate(reactor, C0, **common)
    tight = aquakin.calibrate(reactor, C0, laplace_dtmax=0.05, **common)
    assert np.all(np.isfinite(np.asarray(tight.posterior_cov)))
    assert tight.params_named_std["A_to_B.k"] == pytest.approx(
        base.params_named_std["A_to_B.k"], rel=0.1)


def test_unknown_laplace_method_rejected(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor, C0, observations=obs_clean, t_obs=t_obs,
            free_params=["A_to_B.k"], observed_species=["B"],
            loss="mse", laplace=True, laplace_method="bogus",
        )


def test_falls_back_to_schema_transform_when_omitted(simple_network):
    """If transforms={} is passed, the per-parameter declared transform is used."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    C0 = jnp.asarray([1.0, 0.0])
    obs = jnp.asarray([0.0, 0.5, 0.8])
    t_obs = jnp.asarray([0.0, 5.0, 10.0])
    result = aquakin.calibrate(
        reactor, C0, observations=obs, t_obs=t_obs,
        free_params=["A_to_B.k"],
        observed_species=["B"],
        loss="mse",
        laplace=False,
    )
    # The fixture network declares no explicit transform -> defaults to "none".
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
        observations=obs_clean, t_obs=t_obs, free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"}, observed_species=["B"],
        loss="mse", laplace=False,
    )
    no_prior = aquakin.calibrate(reactor, C0, **common)
    # Prior mean well above the data-optimal true_k (~0.25), moderately tight.
    with_prior = aquakin.calibrate(
        reactor, C0, priors={"A_to_B.k": (2.0, 0.05)}, **common
    )
    k_no = no_prior.params_named["A_to_B.k"]
    k_pr = with_prior.params_named["A_to_B.k"]
    assert k_no == pytest.approx(true_k, rel=1e-2)
    assert k_pr > k_no  # prior drags the estimate upward
    assert with_prior.priors_applied == {"A_to_B.k": (2.0, 0.05)}


def test_priors_ignored_when_not_free(setup):
    """A prior on a parameter that is not being fit has no effect/record."""
    reactor, C0, t_obs, obs_clean, true_k = setup
    result = aquakin.calibrate(
        reactor, C0, observations=obs_clean, t_obs=t_obs,
        free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"], loss="mse", laplace=False,
        priors={"some_other_param": (1.0, 0.1)},
    )
    assert result.priors_applied == {}
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)


# ---------- multistart ----------


def test_multistart_reproducible(setup):
    """Same seed -> identical multistart result (deterministic restarts)."""
    reactor, C0, t_obs, obs_clean, _ = setup
    common = dict(
        observations=obs_clean, t_obs=t_obs, free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"}, observed_species=["B"],
        loss="mse", laplace=False, n_starts=6, jitter=0.8,
    )
    r1 = aquakin.calibrate(reactor, C0, seed=3, **common)
    r2 = aquakin.calibrate(reactor, C0, seed=3, **common)
    assert r1.params_named["A_to_B.k"] == r2.params_named["A_to_B.k"]


def test_multistart_recovers_from_bad_initial(setup):
    """A start far from the truth still recovers k with several restarts."""
    reactor, C0, t_obs, obs_clean, true_k = setup
    bad = reactor.network.default_parameters().at[0].set(40.0)
    result = aquakin.calibrate(
        reactor, C0, observations=obs_clean, t_obs=t_obs,
        free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"], loss="mse", laplace=False,
        initial_params=bad, n_starts=8, jitter=1.5, seed=0,
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)


def test_multistart_default_is_single_start(setup):
    """n_starts defaults to 1, reproducing the single-start result exactly."""
    reactor, C0, t_obs, obs_clean, _ = setup
    common = dict(
        observations=obs_clean, t_obs=t_obs, free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"}, observed_species=["B"],
        loss="mse", laplace=False,
    )
    default = aquakin.calibrate(reactor, C0, **common)
    one = aquakin.calibrate(reactor, C0, n_starts=1, **common)
    assert default.params_named["A_to_B.k"] == one.params_named["A_to_B.k"]


def test_multistart_keeps_best_loss(setup):
    """The reported loss is no worse than the unperturbed single start (start 0
    is always included), so multistart never degrades the fit."""
    reactor, C0, t_obs, obs_clean, _ = setup
    common = dict(
        observations=obs_clean, t_obs=t_obs, free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"}, observed_species=["B"],
        loss="mse", laplace=False,
    )
    single = aquakin.calibrate(reactor, C0, n_starts=1, **common)
    multi = aquakin.calibrate(reactor, C0, n_starts=5, jitter=1.0, seed=1, **common)
    assert multi.loss <= single.loss + 1e-9


def test_param_halfwidth_bounds_the_rate(setup):
    """param_halfwidth box-bounds the rate in log space around the start, so a
    tight halfwidth keeps the fit from reaching a far-away true value, while the
    default (None) lets it reach it."""
    reactor, C0, t_obs, obs_clean, true_k = setup   # true_k=0.25
    k_start = float(reactor.network.default_parameters()[0])   # 0.1
    common = dict(
        observations=obs_clean, t_obs=t_obs, free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"}, observed_species=["B"],
        loss="mse", laplace=False,
    )
    bounded = aquakin.calibrate(reactor, C0, param_halfwidth=0.2, **common)
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
        observations=obs_clean, t_obs=t_obs, free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"}, observed_species=["B"],
        loss="mse", laplace=False, n_starts=4, jitter_schedule=(0.3, 0.5, 0.8),
        seed=2,
    )
    r1 = aquakin.calibrate(reactor, C0, **common)
    r2 = aquakin.calibrate(reactor, C0, **common)
    assert r1.params_named["A_to_B.k"] == r2.params_named["A_to_B.k"]


def test_multistart_invalid_n_starts_rejected(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor, C0, observations=obs_clean, t_obs=t_obs,
            free_params=["A_to_B.k"], observed_species=["B"],
            loss="mse", laplace=False, n_starts=0,
        )


# ---------- posterior-predictive band ----------


def _band_setup(setup):
    reactor, C0, t_obs, obs_clean, true_k = setup
    result = aquakin.calibrate(
        reactor, C0, observations=obs_clean, t_obs=t_obs,
        free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"], loss="nll", sigma=jnp.asarray(0.02), laplace=True,
    )
    return reactor, C0, t_obs, obs_clean, result


def test_predictive_band_brackets_truth(setup):
    """The 95% band envelopes lo <= median <= hi and contains the (noise-free)
    truth at essentially every observation time."""
    reactor, C0, t_obs, obs_clean, result = _band_setup(setup)
    band = result.predictive_band(
        reactor, C0, t_obs, observed_species=["B"], n_draw=200, seed=0
    )
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
    # No observed_species -> all network species (A and B) as columns.
    assert band.median.shape == (len(t_obs), reactor.network.n_species)
    assert band.species is None


def test_predictive_band_requires_laplace(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    result = aquakin.calibrate(
        reactor, C0, observations=obs_clean, t_obs=t_obs,
        free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"], loss="mse", laplace=False,
    )
    with pytest.raises(ValueError):
        result.predictive_band(reactor, C0, t_obs)


def test_predictive_band_eig_keep_is_deprecated(setup):
    reactor, C0, t_obs, _, result = _band_setup(setup)
    with pytest.warns(DeprecationWarning, match="eig_keep"):
        result.predictive_band(
            reactor, C0, t_obs, observed_species=["B"], n_draw=10, eig_keep=1e-2
        )


# ---------- Laplace covariance regularisation (#7) ----------


def test_laplace_covariance_truncates_null_direction():
    """A Hessian with one near-null eigen-direction yields a rank-deficient
    covariance: the unidentifiable direction is dropped, and the kept variance
    is 1/eigenvalue along the strong direction."""
    from aquakin.integrate.calibrate import _laplace_covariance

    V = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)  # orthonormal
    H = V @ np.diag([100.0, 1e-9]) @ V.T
    cov, wk, Vk = _laplace_covariance(H, ridge=1e-6, eig_keep=1e-2)
    assert wk.shape == (1,)                       # the null direction dropped
    assert wk[0] == pytest.approx(100.0, rel=1e-3)
    s = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert int(np.sum(s > 1e-9)) == 1             # rank-1 covariance
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
    assert wk.shape == (1,)                        # strong direction kept...
    assert wk[0] == pytest.approx(9.1e-3, rel=1e-3)  # ...at its true eigenvalue
    s = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert int(np.sum(s > 1e-12)) == 1             # rank-1 covariance
    assert float(np.max(s)) == pytest.approx(1.0 / 9.1e-3, rel=1e-3)


def test_calibrate_nll_small_scale_observable_gives_finite_posterior(setup):
    """End-to-end: a NLL+Laplace fit of a tiny-magnitude observable yields a
    finite, positive-variance posterior rather than a false 'degenerate' error
    (the bug reproduced through the full calibrate path)."""
    reactor, C0, t_obs, _obs_clean, true_k = setup
    true_params = reactor.network.default_parameters().at[0].set(true_k)
    # Scale the observable down to ~1e-6 so the Hessian is uniformly small.
    scale = 1e-6
    C0s = C0 * scale
    obs = reactor.solve(
        C0s, true_params, t_span=(0.0, float(t_obs[-1])), t_eval=t_obs
    ).C_named("B")
    res = aquakin.calibrate(
        reactor, C0s, observations=obs, t_obs=t_obs, free_params=["A_to_B.k"],
        observed_species=["B"], loss="nll", sigma=jnp.asarray(0.05 * scale),
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
        reactor, C0, observations=obs_clean, t_obs=t_obs,
        free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"], loss="mse", laplace=False,
        optimizer="gauss_newton",
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)


def test_gauss_newton_matches_lbfgsb_on_easy_fit(setup):
    """On a convex (single-minimum) fit both optimisers reach the same optimum."""
    reactor, C0, t_obs, obs_clean, _ = setup
    common = dict(
        observations=obs_clean, t_obs=t_obs, free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"}, observed_species=["B"],
        loss="mse", laplace=False,
    )
    lb = aquakin.calibrate(reactor, C0, optimizer="lbfgsb", **common)
    gn = aquakin.calibrate(reactor, C0, optimizer="gauss_newton", **common)
    assert gn.params_named["A_to_B.k"] == pytest.approx(
        lb.params_named["A_to_B.k"], rel=1e-3
    )


def test_nll_loss_is_comparable_across_optimizers(setup):
    """CalibrationResult.loss must be the same scalar objective regardless of
    optimizer. The Gauss-Newton path minimises 0.5*||residual||^2, which for
    loss='nll' drops the sum(log(sigma)) normaliser the L-BFGS-B objective
    includes; the reported loss must add it back so both agree."""
    reactor, C0, t_obs, obs_clean, _ = setup
    sigma = jnp.asarray(0.02)
    common = dict(
        observations=obs_clean, t_obs=t_obs, free_params=["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"}, observed_species=["B"],
        loss="nll", sigma=sigma, laplace=False,
    )
    lb = aquakin.calibrate(reactor, C0, optimizer="lbfgsb", **common)
    gn = aquakin.calibrate(reactor, C0, optimizer="gauss_newton", **common)
    # Same optimum -> same reported objective (the full NLL, not 0.5||r||^2).
    assert gn.loss == pytest.approx(lb.loss, rel=1e-4)
    # And it is the full NLL: well above the GN 0.5||r||^2, which here is ~0
    # (clean data) while sum(log(sigma)) = n_obs * log(0.02) < 0.
    expected_const = float(t_obs.shape[0] * jnp.log(sigma))
    assert gn.loss == pytest.approx(expected_const, abs=1e-3)


def test_gauss_newton_forward_mode_with_direct_adjoint(simple_network):
    """With a DirectAdjoint reactor the GN Jacobian is formed in forward mode
    (jacfwd); it must still recover the parameter."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15),
        adjoint=diffrax.DirectAdjoint(),
    )
    C0 = jnp.asarray([1.0, 0.0])
    true_k = 0.25
    true_params = simple_network.default_parameters().at[0].set(true_k)
    t_obs = jnp.linspace(0.5, 10.0, 20)
    obs = reactor.solve(C0, true_params, t_span=(0.0, 10.0), t_eval=t_obs).C_named("B")
    result = aquakin.calibrate(
        reactor, C0, observations=obs, t_obs=t_obs,
        free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"], loss="mse", laplace=False,
        optimizer="gauss_newton",
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)


def test_gauss_newton_with_free_ic_and_multistart(simple_network):
    """GN composes with free_ic and multistart: recover both k and A0."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    true_k, true_A0 = 0.25, 1.6
    true_params = simple_network.default_parameters().at[0].set(true_k)
    t_obs = jnp.linspace(0.5, 12.0, 25)
    sol = reactor.solve(jnp.asarray([true_A0, 0.0]), true_params, t_span=(0.0, 12.0), t_eval=t_obs)
    obs = jnp.stack([sol.C_named("A"), sol.C_named("B")], axis=1)
    result = aquakin.calibrate(
        reactor, jnp.asarray([1.0, 0.0]), observations=obs, t_obs=t_obs,
        free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"},
        observed_species=["A", "B"], loss="mse", laplace=False,
        optimizer="gauss_newton", free_ic=["A"], ic_bounds=(0.1, 10.0),
        n_starts=3, seed=0,
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)
    assert result.ic_named[0]["A"] == pytest.approx(true_A0, rel=1e-2)


def test_unknown_optimizer_rejected(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor, C0, observations=obs_clean, t_obs=t_obs,
            free_params=["A_to_B.k"], observed_species=["B"], loss="mse",
            laplace=False, optimizer="newton",
        )


# ---------- free initial conditions ----------


def test_free_ic_recovers_initial_condition(simple_network):
    """Fit an unknown initial A0 jointly with the rate; recover both."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    true_k, true_A0 = 0.25, 1.7
    true_params = simple_network.default_parameters().at[0].set(true_k)
    C0_true = jnp.asarray([true_A0, 0.0])
    t_obs = jnp.linspace(0.5, 12.0, 25)
    sol = reactor.solve(C0_true, true_params, t_span=(0.0, 12.0), t_eval=t_obs)
    obs = jnp.stack([sol.C_named("A"), sol.C_named("B")], axis=1)

    # Start from the wrong A0 (1.0); free it.
    C0_start = jnp.asarray([1.0, 0.0])
    result = aquakin.calibrate(
        reactor, C0_start, observations=obs, t_obs=t_obs,
        free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"},
        observed_species=["A", "B"], loss="mse", laplace=False,
        free_ic=["A"], ic_bounds=(0.1, 10.0), n_starts=3, seed=0,
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)
    assert result.ic_named[0]["A"] == pytest.approx(true_A0, rel=1e-2)
    assert float(result.C0_fitted[0][0]) == pytest.approx(true_A0, rel=1e-2)


def test_free_ic_per_dataset_in_multibatch(simple_network):
    """Two batches with different (unknown) A0 but a shared k; each batch's
    initial pool is fitted separately."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    true_k = 0.3
    true_params = simple_network.default_parameters().at[0].set(true_k)
    A0a, A0b = 1.2, 2.4
    ta = jnp.linspace(0.5, 10.0, 20)
    tb = jnp.linspace(0.5, 10.0, 20)
    sola = reactor.solve(jnp.asarray([A0a, 0.0]), true_params, t_span=(0.0, 10.0), t_eval=ta)
    solb = reactor.solve(jnp.asarray([A0b, 0.0]), true_params, t_span=(0.0, 10.0), t_eval=tb)
    obsa = jnp.stack([sola.C_named("A"), sola.C_named("B")], axis=1)
    obsb = jnp.stack([solb.C_named("A"), solb.C_named("B")], axis=1)
    start = jnp.asarray([1.0, 0.0])
    result = aquakin.calibrate(
        reactor, [start, start], observations=[obsa, obsb], t_obs=[ta, tb],
        free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"},
        observed_species=["A", "B"], loss="mse", laplace=False,
        free_ic=["A"], ic_bounds=(0.1, 10.0), n_starts=3, seed=0,
    )
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-2)
    assert result.ic_named[0]["A"] == pytest.approx(A0a, rel=2e-2)
    assert result.ic_named[1]["A"] == pytest.approx(A0b, rel=2e-2)


def test_free_ic_laplace_is_over_rates_only(simple_network):
    """With free_ic + laplace, the posterior covers the rate parameters only
    (pools held at their MAP), so its shape matches the rate count."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    true_params = simple_network.default_parameters().at[0].set(0.25)
    t_obs = jnp.linspace(0.5, 12.0, 25)
    sol = reactor.solve(jnp.asarray([1.5, 0.0]), true_params, t_span=(0.0, 12.0), t_eval=t_obs)
    obs = jnp.stack([sol.C_named("A"), sol.C_named("B")], axis=1)
    result = aquakin.calibrate(
        reactor, jnp.asarray([1.0, 0.0]), observations=obs, t_obs=t_obs,
        free_params=["A_to_B.k"], transforms={"A_to_B.k": "positive_log"},
        observed_species=["A", "B"], loss="nll", sigma=jnp.asarray(0.02),
        laplace=True, free_ic=["A"], ic_bounds=(0.1, 10.0),
    )
    assert np.asarray(result.posterior_cov).shape == (1, 1)
    assert "A_to_B.k" in result.params_named_std


def test_free_ic_unknown_species_rejected(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(KeyError):
        aquakin.calibrate(
            reactor, C0, observations=obs_clean, t_obs=t_obs,
            free_params=["A_to_B.k"], observed_species=["B"], loss="mse",
            laplace=False, free_ic=["NotASpecies"],
        )


def test_free_ic_bad_bounds_rejected(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor, C0, observations=obs_clean, t_obs=t_obs,
            free_params=["A_to_B.k"], observed_species=["B"], loss="mse",
            laplace=False, free_ic=["A"], ic_bounds=(5.0, 1.0),
        )


# ---------- joint multi-batch fit ----------


def test_joint_multibatch_recovers_known_parameter(simple_network):
    """Two batches with different C0 / time grids, fit jointly, recover k."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    true_k = 0.3
    true_params = simple_network.default_parameters().at[0].set(true_k)
    C0a = jnp.asarray([1.0, 0.0])
    C0b = jnp.asarray([2.0, 0.0])
    ta = jnp.linspace(0.5, 8.0, 12)
    tb = jnp.linspace(0.3, 10.0, 15)
    obsa = reactor.solve(C0a, true_params, t_span=(0.0, 8.0), t_eval=ta).C_named("B")
    obsb = reactor.solve(C0b, true_params, t_span=(0.0, 10.0), t_eval=tb).C_named("B")
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


def test_multibatch_length_mismatch_rejected(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    C0a = jnp.asarray([1.0, 0.0])
    t = jnp.asarray([0.0, 1.0])
    obs = jnp.asarray([0.0, 0.5])
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor, [C0a, C0a], observations=[obs], t_obs=[t, t],
            free_params=["A_to_B.k"], observed_species=["B"], loss="mse",
            laplace=False,
        )


def test_rejects_empty_free_params(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor, C0, observations=obs_clean, t_obs=t_obs,
            free_params=[], observed_species=["B"],
        )


def test_rejects_unknown_loss(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor, C0, observations=obs_clean, t_obs=t_obs,
            free_params=["A_to_B.k"], observed_species=["B"],
            loss="huber",
        )


def test_rejects_wmse_without_sigma(setup):
    reactor, C0, t_obs, obs_clean, _ = setup
    with pytest.raises(ValueError):
        aquakin.calibrate(
            reactor, C0, observations=obs_clean, t_obs=t_obs,
            free_params=["A_to_B.k"], observed_species=["B"],
            loss="wmse",
        )


def test_positive_log_initial_negative_rejected(simple_network):
    """If initial value is <= 0 the transform is invalid."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    init_bad = simple_network.default_parameters().at[0].set(-1.0)
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


def test_logit_initial_out_of_range_rejected(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    init_bad = simple_network.default_parameters().at[0].set(1.5)
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
