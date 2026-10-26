"""Integration tests for :func:`aquakin.fisher_information`.

First-order decay ``A -> B`` observed through ``A(t) = A0 exp(-k t)`` has a
closed-form Fisher information, so the AD sensitivities can be checked exactly.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import Experiment, FreeICConfig, LaplaceConfig, OptimizerConfig

K, A0, SIGMA = 0.25, 1.5, 0.01
T_OBS = jnp.linspace(0.5, 12.0, 25)


@pytest.fixture
def reactor(simple_model):
    return aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))


@pytest.fixture
def params(simple_model):
    return simple_model.default_parameters().at[0].set(K)


def _closed_form(t_obs, a0=A0):
    """Fisher information of A(t) in (log k, log A0)."""
    t = np.asarray(t_obs)
    a = a0 * np.exp(-K * t)
    S = np.stack([-K * t * a, a], axis=1)
    return S.T @ S / SIGMA**2


def _design(t_obs=T_OBS, a0=A0, **kw):
    return Experiment(jnp.asarray([a0, 0.0]), t_obs, sigma=SIGMA, observed_species=["A"], **kw)


def test_matches_closed_form_joint_rate_and_pool(reactor, params):
    fim = aquakin.fisher_information(
        reactor,
        _design(),
        ["A_to_B.k"],
        params=params,
        transforms={"A_to_B.k": "positive_log"},
        free_ic=FreeICConfig(["A"]),
    )
    assert fim.names == ["A_to_B.k", "ic[0].A"]
    assert fim.transforms == ["positive_log", "positive_log"]
    np.testing.assert_allclose(fim.matrix, _closed_form(T_OBS), rtol=1e-5)
    np.testing.assert_allclose(fim.jacobian.T @ fim.jacobian, fim.matrix, rtol=1e-12)
    assert fim.values == pytest.approx({"A_to_B.k": K, "ic[0].A": A0})


def test_rate_only_and_physical_space(reactor, params):
    """With no transform the matrix is already physical; ``physical()`` of the
    log-space matrix recovers it, and ``std`` is the Cramer-Rao bound."""
    kw = dict(params=params)
    phys = aquakin.fisher_information(
        reactor, _design(), ["A_to_B.k"], transforms={"A_to_B.k": "none"}, **kw
    )
    logk = aquakin.fisher_information(
        reactor, _design(), ["A_to_B.k"], transforms={"A_to_B.k": "positive_log"}, **kw
    )
    t = np.asarray(T_OBS)
    expected = np.sum((-t * A0 * np.exp(-K * t)) ** 2) / SIGMA**2
    np.testing.assert_allclose(phys.matrix, [[expected]], rtol=1e-5)
    np.testing.assert_allclose(logk.physical(), phys.matrix, rtol=1e-10)
    assert logk.std()["A_to_B.k"] == pytest.approx(expected**-0.5, rel=1e-5)
    assert logk.relative_std()["A_to_B.k"] == pytest.approx(expected**-0.5 / K, rel=1e-5)


def test_matches_calibrate_gauss_newton_hessian_at_map(reactor, params, simple_model):
    """At a calibration's MAP the design information equals the joint
    Gauss-Newton Laplace Hessian (loss='nll')."""
    obs = reactor.solve(
        jnp.asarray([A0, 0.0]), params=params, t_span=(0.0, 12.0), t_eval=T_OBS
    ).C_named("A")
    result = aquakin.calibrate(
        reactor,
        jnp.asarray([1.0, 0.0]),
        obs,
        T_OBS,
        ["A_to_B.k"],
        transforms={"A_to_B.k": "positive_log"},
        observed_species=["A"],
        loss="nll",
        sigma=jnp.asarray(SIGMA),
        free_ic=FreeICConfig(["A"], bounds=(0.1, 10.0)),
        laplace=LaplaceConfig(method="gauss_newton", include_ic=True),
        optimizer=OptimizerConfig(method="gauss_newton"),
    )
    fim = aquakin.fisher_information(
        reactor,
        Experiment(result.C0_fitted[0], T_OBS, sigma=SIGMA, observed_species=["A"]),
        ["A_to_B.k"],
        params=result.params,
        transforms={"A_to_B.k": "positive_log"},
        free_ic=FreeICConfig(["A"]),
    )
    assert fim.names == result.joint_names
    np.testing.assert_allclose(fim.matrix, np.asarray(result.joint_hessian), rtol=1e-6)


def test_experiments_add_and_pools_are_per_experiment(reactor, params):
    """Two experiments: the rate block sums, each pool couples only to its own
    experiment, and the pools are evaluated at each experiment's C0."""
    t2 = jnp.linspace(1.0, 6.0, 10)
    fim = aquakin.fisher_information(
        reactor,
        [_design(), _design(t_obs=t2, a0=0.8)],
        ["A_to_B.k"],
        params=params,
        transforms={"A_to_B.k": "positive_log"},
        free_ic=FreeICConfig(["A"]),
    )
    F1, F2 = _closed_form(T_OBS), _closed_form(t2, a0=0.8)
    expected = np.array(
        [
            [F1[0, 0] + F2[0, 0], F1[0, 1], F2[0, 1]],
            [F1[1, 0], F1[1, 1], 0.0],
            [F2[1, 0], 0.0, F2[1, 1]],
        ]
    )
    assert fim.names == ["A_to_B.k", "ic[0].A", "ic[1].A"]
    np.testing.assert_allclose(fim.matrix, expected, rtol=1e-5, atol=1e-8)
    assert fim.values["ic[1].A"] == pytest.approx(0.8)


def test_per_experiment_reactor_and_time_unit(reactor, params, simple_model):
    """An experiment can carry its own reactor, and t_obs in another unit is
    converted to the model's native unit."""
    other = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=300.0))
    native = aquakin.fisher_information(reactor, _design(), ["A_to_B.k"], params=params)
    unit = simple_model.time_unit
    scaled = aquakin.fisher_information(
        reactor,
        _design(
            t_obs=T_OBS / aquakin.integrate._common.native_time_factor(unit, "min"),
            time_unit="min",
            reactor=other,
        ),
        ["A_to_B.k"],
        params=params,
    )
    np.testing.assert_allclose(scaled.matrix, native.matrix, rtol=1e-6)


def test_priors_add_precision(reactor, params):
    data = aquakin.fisher_information(
        reactor,
        _design(),
        ["A_to_B.k"],
        params=params,
        transforms={"A_to_B.k": "positive_log"},
        free_ic=FreeICConfig(["A"], prior_log_std=0.5),
    )
    with_prior = aquakin.fisher_information(
        reactor,
        _design(),
        ["A_to_B.k"],
        params=params,
        transforms={"A_to_B.k": "positive_log"},
        free_ic=FreeICConfig(["A"], prior_log_std=0.5),
        priors={"A_to_B.k": (0.3, 0.1)},
        use_priors=True,
    )
    # The rate prior is Gaussian in physical k, so in log k it adds (k/0.1)^2.
    added = np.diag([(K / 0.1) ** 2, 1.0 / 0.5**2])
    np.testing.assert_allclose(with_prior.matrix - data.matrix, added, rtol=1e-6, atol=1e-8)


def test_singular_design_raises_on_covariance(reactor, params):
    """A single sample cannot separate k from A0: F is rank one."""
    fim = aquakin.fisher_information(
        reactor,
        _design(t_obs=jnp.asarray([2.0])),
        ["A_to_B.k"],
        params=params,
        transforms={"A_to_B.k": "positive_log"},
        free_ic=FreeICConfig(["A"]),
    )
    assert np.linalg.matrix_rank(fim.matrix) == 1
    with pytest.raises(np.linalg.LinAlgError):
        fim.std()


def test_input_validation(reactor, params):
    with pytest.raises(KeyError, match="Unknown parameter"):
        aquakin.fisher_information(reactor, _design(), ["nope.k"])
    with pytest.raises(KeyError, match="observed species"):
        aquakin.fisher_information(
            reactor,
            Experiment(jnp.asarray([1.0, 0.0]), T_OBS, sigma=SIGMA, observed_species=["Z"]),
            ["A_to_B.k"],
        )
    with pytest.raises(ValueError, match=r"must be\s+positive"):
        aquakin.fisher_information(reactor, _design(), ["A_to_B.k"], free_ic=FreeICConfig(["B"]))
    with pytest.raises(ValueError, match="strictly ascending"):
        aquakin.fisher_information(reactor, _design(t_obs=jnp.asarray([2.0, 1.0])), ["A_to_B.k"])
    with pytest.raises(ValueError, match="at least one"):
        aquakin.fisher_information(reactor, [], ["A_to_B.k"])
