"""Integration tests for :func:`aquakin.fisher_information_from_fn`.

The function-based information is checked against closed forms (straight-line
regression, first-order decay on the log scale) and against
:func:`aquakin.fisher_information` for a batch reactor written as a function.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import Experiment, FreeICConfig

K, A0, SIGMA = 0.25, 1.5, 0.01
T_OBS = jnp.linspace(0.5, 12.0, 25)


def _decay(theta):
    k, a0 = theta
    return a0 * jnp.exp(-k * T_OBS)


def test_linear_regression_closed_form():
    """For ``y = X theta`` the information is ``X^T X / sigma^2`` exactly."""
    x = jnp.linspace(-1.0, 1.0, 7)
    X = np.stack([np.ones(7), np.asarray(x)], axis=1)
    fim = aquakin.fisher_information_from_fn(
        lambda th: th[0] + th[1] * x, jnp.array([0.3, -2.0]), names=["a", "b"], sigma=0.5
    )
    np.testing.assert_allclose(fim.matrix, X.T @ X / 0.25, rtol=1e-12)
    np.testing.assert_allclose(fim.jacobian, X / 0.5, rtol=1e-12)
    assert fim.names == ["a", "b"]
    assert fim.transforms == ["none", "none"]
    assert fim.values == pytest.approx({"a": 0.3, "b": -2.0})


def test_matches_fisher_information_for_batch_reactor(simple_model):
    """A batch reactor written as ``fn(theta)`` gives the same information as
    :func:`aquakin.fisher_information`, rate and pool in log space."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    base = simple_model.default_parameters()
    k_idx = simple_model.param_index["A_to_B.k"]

    def fn(theta):
        sol = reactor.solve(
            jnp.stack([theta[1], 0.0]),
            params=base.at[k_idx].set(theta[0]),
            t_span=(0.0, 12.0),
            t_eval=T_OBS,
        )
        return sol.C_named("A")

    fim = aquakin.fisher_information_from_fn(
        fn,
        jnp.array([K, A0]),
        names=["A_to_B.k", "ic[0].A"],
        sigma=SIGMA,
        transforms={"A_to_B.k": "positive_log", "ic[0].A": "positive_log"},
        mode="reverse",
    )
    ref = aquakin.fisher_information(
        reactor,
        Experiment(jnp.asarray([A0, 0.0]), T_OBS, sigma=SIGMA, observed_species=["A"]),
        ["A_to_B.k"],
        params=base.at[k_idx].set(K),
        transforms={"A_to_B.k": "positive_log"},
        free_ic=FreeICConfig(["A"]),
    )
    assert fim.names == ref.names
    assert fim.transforms == ref.transforms
    np.testing.assert_allclose(fim.matrix, ref.matrix, rtol=1e-6)
    np.testing.assert_allclose(fim.std()["A_to_B.k"], ref.std()["A_to_B.k"], rtol=1e-6)


def test_lognormal_matches_hand_computed():
    """``noise="lognormal"`` observes ``log y``: here ``log A0 - k t``, whose
    Jacobian in ``(k, A0)`` is ``[-t, 1/A0]``."""
    sig_log = 0.05
    fim = aquakin.fisher_information_from_fn(
        _decay, jnp.array([K, A0]), sigma=sig_log, noise="lognormal"
    )
    t = np.asarray(T_OBS)
    J = np.stack([-t, np.full_like(t, 1.0 / A0)], axis=1)
    np.testing.assert_allclose(fim.matrix, J.T @ J / sig_log**2, rtol=1e-10)
    with pytest.raises(ValueError, match="positive"):
        aquakin.fisher_information_from_fn(
            lambda th: th[0] * jnp.array([1.0, -1.0]), jnp.array([1.0]), noise="lognormal"
        )


def test_transforms_and_physical_space():
    """The log-space matrix maps back to the physical one through ``physical()``."""
    theta = jnp.array([K, A0])
    phys = aquakin.fisher_information_from_fn(_decay, theta, sigma=SIGMA)
    logs = aquakin.fisher_information_from_fn(
        _decay, theta, sigma=SIGMA, transforms=["positive_log", "positive_log"]
    )
    np.testing.assert_allclose(logs.physical(), phys.matrix, rtol=1e-10)
    np.testing.assert_allclose(logs.std()["theta[0]"], phys.std()["theta[0]"], rtol=1e-10)


def test_forward_and_reverse_agree():
    theta = jnp.array([K, A0])
    fwd = aquakin.fisher_information_from_fn(_decay, theta, sigma=SIGMA, mode="forward")
    rev = aquakin.fisher_information_from_fn(_decay, theta, sigma=SIGMA, mode="reverse")
    np.testing.assert_allclose(fwd.matrix, rev.matrix, rtol=1e-12)


def test_per_observation_sigma_and_unmeasured_entries():
    """``sigma`` broadcasts to the observations; ``inf`` drops an entry."""
    x = jnp.linspace(0.0, 1.0, 4)
    sig = np.array([1.0, 2.0, np.inf, 0.5])
    fim = aquakin.fisher_information_from_fn(lambda th: th[0] * x, jnp.array([1.0]), sigma=sig)
    expected = np.sum(np.where(np.isinf(sig), 0.0, np.asarray(x) ** 2 / sig**2))
    np.testing.assert_allclose(fim.matrix, [[expected]], rtol=1e-12)


def test_groups_split_the_information():
    """Each group's information is that of its own rows, labels keep their
    order of first appearance, and the groups sum to the total."""
    theta = jnp.array([0.4, 1.2, -0.3])

    def fn(th):
        t = jnp.linspace(0.0, 2.0, 5)
        # Three "sites" (rows) by five times (columns).
        return jnp.stack([th[0] + th[1] * t, th[1] * t**2, th[2] + th[0] * t])

    total, parts = aquakin.fisher_information_from_fn(
        fn, theta, names=["a", "b", "c"], sigma=0.1, groups=["s2", "s1", "s2"]
    )
    assert list(parts) == ["s2", "s1"]
    np.testing.assert_allclose(sum(f.matrix for f in parts.values()), total.matrix, rtol=1e-12)
    only_s1 = aquakin.fisher_information_from_fn(lambda th: fn(th)[1], theta, sigma=0.1)
    np.testing.assert_allclose(parts["s1"].matrix, only_s1.matrix, rtol=1e-12)
    assert parts["s1"].jacobian.shape == (5, 3)
    assert parts["s1"].names == ["a", "b", "c"]

    # A label per observation works too, and integer labels are kept as-is.
    per_obs = np.tile(np.arange(5), (3, 1))
    _, by_time = aquakin.fisher_information_from_fn(fn, theta, sigma=0.1, groups=per_obs)
    assert list(by_time) == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(sum(f.matrix for f in by_time.values()), total.matrix, rtol=1e-12)


def test_works_with_identifiability_and_design_criterion():
    """The result feeds the matrix-level tools unchanged."""
    x = jnp.linspace(-1.0, 1.0, 5)
    # ``c`` never enters the observations: one exactly non-identifiable direction.
    flat = aquakin.fisher_information_from_fn(
        lambda th: th[0] + th[1] * x + 0.0 * th[2],
        jnp.array([1.0, 1.0, 1.0]),
        names=["a", "b", "c"],
    )
    report = aquakin.identifiability(flat)
    unidentified = [d for d in report.directions if not d.identifiable]
    assert len(unidentified) == 1
    assert abs(unidentified[0].loadings["c"]) == pytest.approx(1.0)

    quad = aquakin.fisher_information_from_fn(
        lambda th: th[0] + th[1] * x + th[2] * x**2,
        jnp.array([1.0, 1.0, 1.0]),
        names=["a", "b", "c"],
    )
    X = np.stack([np.ones(5), np.asarray(x), np.asarray(x) ** 2], axis=1)
    F = X.T @ X
    schur = F[:2, :2] - np.outer(F[:2, 2], F[2, :2]) / F[2, 2]
    assert aquakin.design_criterion(quad, "D", subset=["a", "b"]) == pytest.approx(
        np.log(np.linalg.det(schur))
    )


def test_input_validation():
    theta = jnp.array([K, A0])
    with pytest.raises(ValueError, match="noise"):
        aquakin.fisher_information_from_fn(_decay, theta, noise="poisson")
    with pytest.raises(ValueError, match="mode"):
        aquakin.fisher_information_from_fn(_decay, theta, mode="sideways")
    with pytest.raises(ValueError, match="names"):
        aquakin.fisher_information_from_fn(_decay, theta, names=["k"])
    with pytest.raises(ValueError, match="1-D"):
        aquakin.fisher_information_from_fn(_decay, jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="broadcast"):
        aquakin.fisher_information_from_fn(_decay, theta, sigma=np.ones(3))
    with pytest.raises(ValueError, match="positive"):
        aquakin.fisher_information_from_fn(_decay, theta, sigma=0.0)
    with pytest.raises(ValueError, match="transforms"):
        aquakin.fisher_information_from_fn(_decay, theta, transforms=["none"])
    with pytest.raises(KeyError, match="Unknown transform name"):
        aquakin.fisher_information_from_fn(_decay, theta, transforms={"q": "none"})
    with pytest.raises(ValueError, match="domain"):
        aquakin.fisher_information_from_fn(
            _decay, jnp.array([-K, A0]), transforms=["positive_log", "none"]
        )
    with pytest.raises(ValueError, match="groups"):
        aquakin.fisher_information_from_fn(_decay, theta, groups=np.zeros(3))
    with pytest.raises(RuntimeError, match="non-finite"):
        aquakin.fisher_information_from_fn(lambda th: jnp.sqrt(th - th[0]), jnp.array([1.0, 2.0]))
