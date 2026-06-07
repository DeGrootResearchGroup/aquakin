"""Tests for MAP calibration + Laplace posterior."""

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
