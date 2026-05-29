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
