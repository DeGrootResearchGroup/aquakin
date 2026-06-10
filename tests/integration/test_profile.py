"""Tests for profile-likelihood identifiability analysis."""

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def growth_setup():
    """Two-parameter growth model (mu, Y) with informative noisy data, so the
    parameters are well identified -- a clean setting for profile-vs-Laplace."""
    net = aquakin.load_network_from_file(str(FIXTURES / "dynamic_stoich_network.yaml"))
    reactor = aquakin.BatchReactor(net, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0 = net.default_concentrations()
    true = net.default_parameters()
    t = jnp.linspace(0.02, 2.0, 40)
    sol = reactor.solve(C0, true, t_span=(0.0, 2.0), t_eval=t)
    rng = np.random.default_rng(1)
    sig = 0.3
    obs = jnp.stack(
        [jnp.asarray(np.asarray(sol.C_named(s)) + sig * rng.standard_normal(t.shape))
         for s in ("S", "X")], axis=1
    )
    return net, reactor, C0, t, obs, sig


def test_profile_param_ci_matches_laplace(growth_setup):
    """On a well-identified, near-Gaussian problem the profile-likelihood 95%
    interval (delta=1.92 crossings) matches the Laplace MAP +/- 1.96 sigma."""
    net, reactor, C0, t, obs, sig = growth_setup
    cal = aquakin.calibrate(
        reactor, C0, observations=obs, t_obs=t, free_params=["mu", "Y"],
        observed_species=["S", "X"], loss="nll", sigma=jnp.asarray(sig),
        laplace=True, laplace_method="gauss_newton",
    )
    mu_hat = cal.params_named["mu"]
    mu_std = cal.params_named_std["mu"]
    grid = np.linspace(max(0.05, mu_hat - 2.6 * mu_std), mu_hat + 2.6 * mu_std, 21)
    pr = aquakin.profile_likelihood(
        reactor, C0, obs, t, ["mu", "Y"], grid=grid, profile_param="mu",
        observed_species=["S", "X"], loss="nll", sigma=jnp.asarray(sig), n_starts=4,
    )
    # Profile minimum == joint MLE.
    assert pr.mle == pytest.approx(mu_hat, rel=2e-2)
    assert np.nanmin(pr.delta_loss) == pytest.approx(0.0, abs=1e-6)
    # Two-sided interval that matches the Laplace 95% bounds.
    lo, hi = pr.ci
    assert lo is not None and hi is not None
    assert lo == pytest.approx(mu_hat - 1.96 * mu_std, abs=0.02)
    assert hi == pytest.approx(mu_hat + 1.96 * mu_std, abs=0.02)
    assert lo < pr.mle < hi


def test_profile_is_a_bowl(growth_setup):
    """delta_loss is ~0 at the minimum and rises away from it."""
    net, reactor, C0, t, obs, sig = growth_setup
    grid = np.linspace(0.6, 1.4, 17)
    pr = aquakin.profile_likelihood(
        reactor, C0, obs, t, ["mu", "Y"], grid=grid, profile_param="mu",
        observed_species=["S", "X"], loss="nll", sigma=jnp.asarray(sig), n_starts=4,
    )
    imin = int(np.nanargmin(pr.delta_loss))
    assert pr.delta_loss[imin] == pytest.approx(0.0, abs=1e-6)
    # Strictly higher at both ends than at the minimum.
    assert pr.delta_loss[0] > 1.0
    assert pr.delta_loss[-1] > 1.0


def test_profile_warmstart_matches_independent(growth_setup):
    """Continuation sweep and independent multistart find the same minimum."""
    net, reactor, C0, t, obs, sig = growth_setup
    grid = np.linspace(0.7, 1.3, 13)
    common = dict(
        profile_param="mu", observed_species=["S", "X"], loss="nll",
        sigma=jnp.asarray(sig), n_starts=4,
    )
    warm = aquakin.profile_likelihood(reactor, C0, obs, t, ["mu", "Y"], grid=grid,
                                      warm_start=True, **common)
    indep = aquakin.profile_likelihood(reactor, C0, obs, t, ["mu", "Y"], grid=grid,
                                       warm_start=False, **common)
    assert warm.mle == pytest.approx(indep.mle, rel=2e-2)
    assert np.allclose(warm.delta_loss, indep.delta_loss, atol=0.05, equal_nan=True)


# ---------- initial-condition profile ----------


def test_profile_ic_recovers_initial_condition(simple_network):
    """Profiling an unmeasured initial A0 (re-optimising the rate at each value)
    locates it at the true value."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    true_k, true_A0 = 0.25, 1.7
    true_params = simple_network.default_parameters().at[0].set(true_k)
    t = jnp.linspace(0.5, 12.0, 25)
    sol = reactor.solve(jnp.asarray([true_A0, 0.0]), true_params, t_span=(0.0, 12.0), t_eval=t)
    obs = jnp.stack([sol.C_named("A"), sol.C_named("B")], axis=1)
    grid = np.linspace(1.3, 2.1, 17)
    pr = aquakin.profile_likelihood(
        reactor, jnp.asarray([1.0, 0.0]), obs, t, ["A_to_B.k"], grid=grid,
        profile_ic="A", transforms={"A_to_B.k": "positive_log"},
        observed_species=["A", "B"], loss="nll", sigma=jnp.asarray(0.02), n_starts=4,
    )
    assert pr.profiled == "A"
    assert pr.mle == pytest.approx(true_A0, abs=0.06)


def test_profile_open_interval_when_unidentifiable(simple_network):
    """Observing only B over an early window, A0 and k are degenerate (only A0*k
    is constrained). Profiling A0 (re-optimising k) leaves the objective flat, so
    the interval is open on at least one side."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    true_params = simple_network.default_parameters().at[0].set(0.25)
    t = jnp.linspace(0.1, 0.6, 8)   # early: B ~ A0*k*t, only the product is seen
    obs = reactor.solve(jnp.asarray([1.5, 0.0]), true_params, t_span=(0.0, 0.6), t_eval=t).C_named("B")
    grid = np.linspace(1.0, 2.2, 13)
    pr = aquakin.profile_likelihood(
        reactor, jnp.asarray([1.5, 0.0]), obs, t, ["A_to_B.k"], grid=grid,
        profile_ic="A", transforms={"A_to_B.k": "positive_log"},
        observed_species=["B"], loss="nll", sigma=jnp.asarray(0.01), n_starts=4,
    )
    # Flat profile -> at least one bound is open (None).
    assert pr.ci[0] is None or pr.ci[1] is None
    assert float(np.nanmax(pr.delta_loss)) < 1.92


# ---------- validation ----------


def test_profile_requires_exactly_one_target(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    args = (reactor, jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 0.5]),
            jnp.asarray([0.0, 1.0]), ["A_to_B.k"])
    with pytest.raises(ValueError):   # neither
        aquakin.profile_likelihood(*args, grid=[0.1, 0.2], observed_species=["B"])
    with pytest.raises(ValueError):   # both
        aquakin.profile_likelihood(*args, grid=[0.1, 0.2], profile_param="A_to_B.k",
                                   profile_ic="A", observed_species=["B"])


def test_profile_only_free_param_rejected(simple_network):
    """Profiling the single free parameter leaves nothing to re-optimise."""
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    with pytest.raises(ValueError):
        aquakin.profile_likelihood(
            reactor, jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 0.5]),
            jnp.asarray([0.0, 1.0]), ["A_to_B.k"], grid=[0.1, 0.2, 0.3],
            profile_param="A_to_B.k", observed_species=["B"],
        )


def test_profile_unknown_target_rejected(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    base = (reactor, jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 0.5]),
            jnp.asarray([0.0, 1.0]), ["A_to_B.k"])
    with pytest.raises(KeyError):
        aquakin.profile_likelihood(*base, grid=[0.1, 0.2], profile_param="nope",
                                   observed_species=["B"])
    with pytest.raises(KeyError):
        aquakin.profile_likelihood(*base, grid=[0.1, 0.2], profile_ic="nope",
                                   observed_species=["B"])


def test_interp_ci_clean_bowl():
    """A symmetric bowl crossing the threshold on both sides gives a finite CI
    and no warning."""
    from aquakin.integrate.profile import _interp_ci

    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    delta_loss = np.array([4.0, 1.0, 0.0, 1.0, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails
        lo, hi = _interp_ci(grid, delta_loss, delta=1.92)
    assert lo is not None and hi is not None
    assert 0.0 < lo < 1.0 and 3.0 < hi < 4.0


def test_interp_ci_nan_in_crossing_region_warns_and_is_open():
    """A failed inner fit (NaN) between the minimum and the lower edge blocks
    that side: it returns None but WARNS, so it is not silently read as
    non-identifiability."""
    from aquakin.integrate.profile import _interp_ci

    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    delta_loss = np.array([4.0, np.nan, 0.0, 1.0, 4.0])  # NaN on the lower side
    with pytest.warns(UserWarning, match="lower confidence bound is indeterminate"):
        lo, hi = _interp_ci(grid, delta_loss, delta=1.92)
    assert lo is None              # blocked by the NaN
    assert hi is not None          # upper side is clean


def test_interp_ci_genuinely_open_does_not_warn():
    """A flat profile that never crosses the threshold is genuinely open and
    must NOT warn (no NaNs in the way)."""
    from aquakin.integrate.profile import _interp_ci

    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    delta_loss = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # below delta everywhere
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lo, hi = _interp_ci(grid, delta_loss, delta=1.92)
    assert lo is None and hi is None


def test_profile_all_failed_returns_unidentified(simple_network, monkeypatch):
    """If every inner fit fails, the profile is all-NaN: return a clean
    'unidentified' result (mle=nan, open CI) rather than raising on nanmin/
    nanargmin of an all-NaN array. Each failure must also surface as a warning
    carrying the exception type (so a config bug is not silently swallowed)."""
    from aquakin.integrate import profile as profile_mod

    def _always_fail(*args, **kwargs):
        raise RuntimeError("inner fit blew up")

    monkeypatch.setattr(profile_mod, "calibrate", _always_fail)
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    grid = np.linspace(0.1, 0.4, 5)
    with pytest.warns(UserWarning, match="inner fit failed.*RuntimeError"):
        pr = aquakin.profile_likelihood(
            reactor, jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 0.5]),
            jnp.asarray([0.0, 1.0]), ["A_to_B.k"], grid=grid,
            profile_ic="A", observed_species=["B"], n_starts=2,
        )
    assert np.isnan(pr.mle)
    assert pr.ci == (None, None)
    assert np.all(np.isnan(pr.loss))
    assert np.all(np.isnan(pr.delta_loss))
    assert all(f is None for f in pr.fits)


def test_profile_multibatch_rejected(simple_network):
    reactor = aquakin.BatchReactor(
        simple_network, aquakin.SpatialConditions.uniform(1, T=293.15)
    )
    C0 = jnp.asarray([1.0, 0.0])
    with pytest.raises(NotImplementedError):
        aquakin.profile_likelihood(
            reactor, [C0, C0], [jnp.asarray([0.0, 0.5])] * 2,
            [jnp.asarray([0.0, 1.0])] * 2, ["A_to_B.k"], grid=[0.1, 0.2],
            profile_ic="A", observed_species=["B"],
        )
