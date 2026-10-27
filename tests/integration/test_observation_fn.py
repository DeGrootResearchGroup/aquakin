"""Integration tests for ``observation_fn``: measurements that are functions of
several states (and possibly parameters), in ``calibrate``,
``profile_likelihood`` and ``fisher_information``.

For ``A -> B``, ``A + B`` is conserved, so the surrogate ``y = 2A + B`` equals
``A + const`` and carries exactly the information about ``k`` that observing
``A`` does.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import Experiment, LaplaceConfig, OptimizerConfig

K, A0, SIGMA = 0.25, 1.5, 0.01
T_OBS = jnp.linspace(0.5, 12.0, 25)
C0 = jnp.asarray([A0, 0.0])


def surrogate(C, p):
    return 2.0 * C[0] + C[1]


@pytest.fixture
def reactor(simple_model):
    return aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))


@pytest.fixture
def params(simple_model):
    return simple_model.default_parameters().at[0].set(K)


@pytest.fixture
def truth(reactor, params):
    return reactor.solve(C0, params=params, t_span=(0.0, 12.0), t_eval=T_OBS).C


def _fit(reactor, observations, **kw):
    return aquakin.calibrate(
        reactor,
        C0,
        observations,
        T_OBS,
        ["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        loss="nll",
        laplace=LaplaceConfig(method="gauss_newton"),
        optimizer=OptimizerConfig(method="gauss_newton"),
        **kw,
    )


def test_calibrate_with_surrogate_matches_observing_a(reactor, truth):
    y = 2.0 * truth[:, 0] + truth[:, 1]
    res = _fit(reactor, y, observation_fn=surrogate, sigma=jnp.asarray(SIGMA))
    ref = _fit(reactor, truth[:, 0], observed_species=["A"], sigma=jnp.asarray(SIGMA))
    assert res.params_named["A_to_B.k"] == pytest.approx(K, rel=1e-5)
    np.testing.assert_allclose(
        np.asarray(res.hessian_unconstrained), np.asarray(ref.hessian_unconstrained), rtol=1e-5
    )


def test_vector_output_with_per_output_sigma(reactor, truth):
    """A function returning [A, B] with a per-output sigma is the same fit as
    observed_species=["A", "B"]."""
    sigma = jnp.asarray([SIGMA, 2 * SIGMA])
    res = _fit(reactor, truth, observation_fn=lambda C, p: C[:2], sigma=sigma)
    ref = _fit(reactor, truth, observed_species=["A", "B"], sigma=sigma)
    np.testing.assert_allclose(
        np.asarray(res.hessian_unconstrained), np.asarray(ref.hessian_unconstrained), rtol=1e-8
    )


def test_profile_likelihood_with_surrogate(reactor, truth):
    """Profile the initial pool of A (k re-fitted at each point) on the surrogate."""
    y = 2.0 * truth[:, 0] + truth[:, 1]
    prof = aquakin.profile_likelihood(
        reactor,
        C0,
        y,
        T_OBS,
        ["A_to_B.k"],
        grid=jnp.asarray([1.3, 1.5, 1.7]),
        profile_ic="A",
        observation_fn=surrogate,
        sigma=jnp.asarray(SIGMA),
        optimizer=OptimizerConfig(n_starts=1),
    )
    assert prof.mle == pytest.approx(A0)


def test_fisher_information_parameter_dependent_observation(reactor, params):
    """y = k A depends on k directly, so dy/dk = A0 exp(-k t) (1 - k t)."""
    fim = aquakin.fisher_information(
        reactor,
        Experiment(C0, T_OBS, sigma=SIGMA, observation_fn=lambda C, p: p[0] * C[0]),
        ["A_to_B.k"],
        params=params,
        transforms={"A_to_B.k": "none"},
    )
    t = np.asarray(T_OBS)
    dy = A0 * np.exp(-K * t) * (1.0 - K * t)
    np.testing.assert_allclose(fim.matrix, [[np.sum(dy**2) / SIGMA**2]], rtol=1e-5)

    surr = aquakin.fisher_information(
        reactor,
        Experiment(C0, T_OBS, sigma=SIGMA, observation_fn=surrogate),
        ["A_to_B.k"],
        params=params,
    )
    ref = aquakin.fisher_information(
        reactor,
        Experiment(C0, T_OBS, sigma=SIGMA, observed_species=["A"]),
        ["A_to_B.k"],
        params=params,
    )
    np.testing.assert_allclose(surr.matrix, ref.matrix, rtol=1e-8)


def test_validation(reactor, truth):
    y = 2.0 * truth[:, 0] + truth[:, 1]
    with pytest.raises(ValueError, match="not both"):
        _fit(reactor, y, observation_fn=surrogate, observed_species=["A"], sigma=SIGMA)
    with pytest.raises(TypeError, match="callable"):
        _fit(reactor, y, observation_fn="2A+B", sigma=SIGMA)
    with pytest.raises(ValueError, match="scalar or 1-D"):
        _fit(reactor, y, observation_fn=lambda C, p: jnp.outer(C, C), sigma=SIGMA)
    with pytest.raises(ValueError, match="outputs were returned by observation_fn"):
        _fit(reactor, truth, observation_fn=surrogate, sigma=SIGMA)
    with pytest.raises(ValueError, match="not both"):
        aquakin.fisher_information(
            reactor,
            Experiment(C0, T_OBS, sigma=SIGMA, observation_fn=surrogate, observed_species=["A"]),
            ["A_to_B.k"],
        )
