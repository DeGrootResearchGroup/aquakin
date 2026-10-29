"""Tests for FreeICConfig per-species bounds and the fixed-total (simplex)
parameterisation of fitted initial pools, in calibrate and fisher_information.

The model has a fast pool S and a slow pool X, both degrading to P. Only P is
measured, so the split of the initial total between S and X is seen through the
two time scales.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import Experiment, FreeICConfig, LaplaceConfig, OptimizerConfig
from aquakin.integrate.calibrate import _free_ic_fields

_TWO_POOL_YAML = """
model:
  name: two_pool
  version: "1.0"
  description: "Fast pool S and slow pool X both degrading to P."
  references: []
species:
  - name: S
    units: g/m3
    default_concentration: 0.5
  - name: X
    units: g/m3
    default_concentration: 0.5
  - name: P
    units: g/m3
    default_concentration: 0.0
reactions:
  - name: fast
    rate: "kS * [S]"
    parameters:
      kS:
        value: 1.0
        bounds: [1.0e-3, 100.0]
        transform: positive_log
    stoichiometry:
      S: -1
      P: +1
  - name: slow
    rate: "kX * [X]"
    parameters:
      kX:
        value: 0.1
        bounds: [1.0e-4, 10.0]
        transform: positive_log
    stoichiometry:
      X: -1
      P: +1
"""

T_OBS = jnp.linspace(0.25, 20.0, 30)
SIGMA = 0.01


@pytest.fixture(scope="module")
def two_pool(tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "two_pool.yaml"
    path.write_text(_TWO_POOL_YAML)
    model = aquakin.load_model_from_file(path)
    reactor = aquakin.BatchReactor(model, aquakin.SpatialConditions.uniform(1))
    return model, reactor


def _data(reactor, S0, X0):
    C0 = jnp.asarray([S0, X0, 0.0])
    sol = reactor.solve(C0, t_span=(0.0, float(T_OBS[-1])), t_eval=T_OBS)
    return sol.C_named("P")


_COMMON = dict(
    t_obs=T_OBS,
    free_params=["fast.kS"],
    observed_species=["P"],
    loss="nll",
    sigma=SIGMA,
    optimizer=OptimizerConfig(method="gauss_newton"),
)


def test_fixed_total_recovers_split_and_keeps_total(two_pool):
    """Data with S0 = 0.3, X0 = 0.7 (total 1.0); the fit starts from the even
    split 0.5 / 0.5 with the total in C0, and recovers the split while the
    pools sum to the total exactly."""
    model, reactor = two_pool
    obs = _data(reactor, 0.3, 0.7)
    r = aquakin.calibrate(
        reactor,
        jnp.asarray([0.5, 0.5, 0.0]),
        obs,
        free_ic=FreeICConfig(["S", "X"], fixed_total=True),
        laplace=LaplaceConfig(method="gauss_newton", include_ic=True),
        **_COMMON,
    )
    assert r.ic_named[0]["S"] == pytest.approx(0.3, rel=1e-4)
    assert r.ic_named[0]["X"] == pytest.approx(0.7, rel=1e-4)
    assert r.ic_named[0]["S"] + r.ic_named[0]["X"] == pytest.approx(1.0, rel=1e-12)
    assert r.ic_fractions[0]["S"] == pytest.approx(0.3, rel=1e-4)
    assert r.params_named["fast.kS"] == pytest.approx(1.0, rel=1e-4)
    # One log-ratio coordinate for two pools.
    assert r.joint_names == ["fast.kS", "ic[0].X:S"]
    # Fractions std: for two pools d f_S / d z = -f_S f_X, so
    # sd(f_S) = sd(f_X) = f_S f_X sd(z), and the pool std is total * sd(f).
    sd_z = float(np.sqrt(np.asarray(r.joint_cov)[1, 1]))
    fS, fX = r.ic_fractions[0]["S"], r.ic_fractions[0]["X"]
    assert r.ic_fractions_std[0]["S"] == pytest.approx(fS * fX * sd_z, rel=1e-9)
    assert r.ic_fractions_std[0]["X"] == pytest.approx(fS * fX * sd_z, rel=1e-9)
    assert r.ic_named_std[0]["S"] == pytest.approx(1.0 * fS * fX * sd_z, rel=1e-9)
    assert 0.0 < r.ic_fractions_std[0]["S"] < 0.05


def test_fixed_total_reproduces_constrained_optimum(two_pool):
    """On noisy data the fixed-total optimum is not the true split. It matches
    a brute-force scan of the split, each point a calibrate fit of kS with the
    split held fixed and the total at 1.0."""
    model, reactor = two_pool
    rng = np.random.default_rng(0)
    obs = _data(reactor, 0.3, 0.7) + 3 * SIGMA * rng.standard_normal(T_OBS.shape)
    r = aquakin.calibrate(
        reactor,
        jnp.asarray([0.5, 0.5, 0.0]),
        obs,
        free_ic=FreeICConfig(["S", "X"], fixed_total=True),
        laplace=False,
        **_COMMON,
    )
    cache: dict = {}
    grid = np.linspace(0.2, 0.4, 41)
    losses = []
    for f in grid:
        fit = aquakin.calibrate(
            reactor,
            jnp.asarray([f, 1.0 - f, 0.0]),
            obs,
            laplace=False,
            _compiled_cache=cache,
            **_COMMON,
        )
        losses.append(fit.loss)
    losses = np.asarray(losses)
    i = int(np.argmin(losses))
    assert 0 < i < len(grid) - 1
    # Refine with the parabola through the three points around the minimum.
    a, b, _c = np.polyfit(grid[i - 1 : i + 2], losses[i - 1 : i + 2], 2)
    f_scan = -b / (2.0 * a)
    assert r.ic_fractions[0]["S"] == pytest.approx(f_scan, abs=1e-3)
    assert abs(r.ic_fractions[0]["S"] - 0.3) > 1e-3  # a genuine compromise
    assert r.loss <= losses.min() + 1e-6
    assert r.ic_named[0]["S"] + r.ic_named[0]["X"] == pytest.approx(1.0, rel=1e-12)


def test_fixed_total_fisher_matches_joint_hessian(two_pool):
    """fisher_information with fixed_total uses the same log-ratio coordinates,
    so at the MAP it equals the joint Gauss-Newton Hessian."""
    model, reactor = two_pool
    obs = _data(reactor, 0.3, 0.7)
    cfg = FreeICConfig(["S", "X"], fixed_total=True)
    r = aquakin.calibrate(
        reactor,
        jnp.asarray([0.5, 0.5, 0.0]),
        obs,
        free_ic=cfg,
        laplace=LaplaceConfig(method="gauss_newton", include_ic=True),
        **_COMMON,
    )
    fim = aquakin.fisher_information(
        reactor,
        Experiment(r.C0_fitted[0], T_OBS, sigma=SIGMA, observed_species=["P"]),
        ["fast.kS"],
        params=r.params,
        free_ic=cfg,
    )
    assert fim.names == r.joint_names
    np.testing.assert_allclose(fim.matrix, np.asarray(r.joint_hessian), rtol=1e-6)
    assert fim.values["ic[0].X:S"] == pytest.approx(r.ic_named[0]["X"] / r.ic_named[0]["S"])
    # identifiability reads the ratio value from the calibration result.
    rep = aquakin.identifiability(r)
    assert np.isfinite(rep.std["ic[0].X:S"])


def test_fixed_total_multi_dataset_uses_each_total(two_pool):
    """Each dataset keeps its own total from its C0."""
    model, reactor = two_pool
    obs_a = _data(reactor, 0.3, 0.7)
    obs_b = _data(reactor, 1.0, 1.0)
    r = aquakin.calibrate(
        reactor,
        [jnp.asarray([0.5, 0.5, 0.0]), jnp.asarray([1.5, 0.5, 0.0])],
        [obs_a, obs_b],
        t_obs=[T_OBS, T_OBS],
        free_params=["fast.kS"],
        observed_species=["P"],
        loss="nll",
        sigma=SIGMA,
        optimizer=OptimizerConfig(method="gauss_newton"),
        free_ic=FreeICConfig(["S", "X"], fixed_total=True),
        laplace=False,
    )
    assert r.ic_named[0]["S"] == pytest.approx(0.3, rel=1e-3)
    assert r.ic_named[1]["S"] == pytest.approx(1.0, rel=1e-3)
    assert r.ic_named[1]["S"] + r.ic_named[1]["X"] == pytest.approx(2.0, rel=1e-12)


def test_per_species_bounds(two_pool):
    """A per-species bound applies to its pool only: S0 (truth 0.3) is capped
    at 0.2, while X keeps the default box and absorbs what it can."""
    model, reactor = two_pool
    obs = _data(reactor, 0.3, 0.7)
    with pytest.warns(RuntimeWarning, match=r"ic\[0\]\.S.*ended on an optimiser bound"):
        r = aquakin.calibrate(
            reactor,
            jnp.asarray([0.1, 0.5, 0.0]),
            obs,
            free_ic=FreeICConfig(["S", "X"], bounds={"S": (0.01, 0.2)}),
            laplace=False,
            **_COMMON,
        )
    assert r.ic_named[0]["S"] == pytest.approx(0.2, rel=1e-6)
    assert r.ic_named[0]["X"] > 0.7
    # The pool on its own per-species bound is flagged; X is not.
    assert r.at_bound == {"ic[0].S": "upper"}
    # The shared-pair form still works and leaves the truth reachable.
    r2 = aquakin.calibrate(
        reactor,
        jnp.asarray([0.1, 0.5, 0.0]),
        obs,
        free_ic=FreeICConfig(["S", "X"], bounds=(0.01, 5.0)),
        laplace=False,
        **_COMMON,
    )
    assert r2.ic_named[0]["S"] == pytest.approx(0.3, rel=1e-3)


def test_free_ic_config_validation(two_pool):
    model, reactor = two_pool
    obs = _data(reactor, 0.3, 0.7)
    C0 = jnp.asarray([0.5, 0.5, 0.0])
    kw = dict(laplace=False, **_COMMON)
    with pytest.raises(KeyError, match="not fitted"):
        aquakin.calibrate(reactor, C0, obs, free_ic=FreeICConfig(["S"], bounds={"X": (1, 2)}), **kw)
    with pytest.raises(ValueError, match="0 < lo < hi"):
        aquakin.calibrate(reactor, C0, obs, free_ic=FreeICConfig(["S"], bounds={"S": (2, 1)}), **kw)
    with pytest.raises(ValueError, match="takes no bounds"):
        aquakin.calibrate(
            reactor,
            C0,
            obs,
            free_ic=FreeICConfig(["S", "X"], bounds=(0.1, 2.0), fixed_total=True),
            **kw,
        )
    with pytest.raises(ValueError, match="must be positive in C0"):
        aquakin.calibrate(
            reactor,
            jnp.asarray([1.0, 0.0, 0.0]),
            obs,
            free_ic=FreeICConfig(["S", "X"], fixed_total=True),
            **kw,
        )
    # The extensions reach every caller (the plant path included) unchanged.
    assert _free_ic_fields(FreeICConfig(["S", "X"], fixed_total=True))[0] == ["S", "X"]
    assert _free_ic_fields(FreeICConfig(["S"], bounds={"S": (1.0, 2.0)}))[1] == {"S": (1.0, 2.0)}


def test_per_species_bound_start_outside_and_laplace(two_pool):
    """A start outside a pool's own bounds begins on that bound; the pool that
    ends on its bound gets a nan std in the joint Laplace posterior and zero
    covariance, while the rate and the free pool keep finite stds."""
    model, reactor = two_pool
    obs = _data(reactor, 0.3, 0.7)
    with pytest.warns(RuntimeWarning, match="ended on an optimiser bound"):
        r = aquakin.calibrate(
            reactor,
            jnp.asarray([0.9, 0.5, 0.0]),  # S starts above its cap 0.2
            obs,
            free_ic=FreeICConfig(["S", "X"], bounds={"S": (0.01, 0.2), "X": (0.05, 5.0)}),
            laplace=LaplaceConfig(method="gauss_newton", include_ic=True),
            **_COMMON,
        )
    assert r.ic_named[0]["S"] == pytest.approx(0.2, rel=1e-6)
    assert r.at_bound == {"ic[0].S": "upper"}
    assert np.isnan(r.ic_named_std[0]["S"])
    assert np.isfinite(r.ic_named_std[0]["X"]) and r.ic_named_std[0]["X"] > 0.0
    assert np.isfinite(r.params_named_std["fast.kS"])
    i = r.joint_names.index("ic[0].S")
    assert np.all(np.asarray(r.joint_cov)[i] == 0.0)
    rep = aquakin.identifiability(r)
    assert rep.at_bound == {"ic[0].S": "upper"}
    assert np.isnan(rep.std["ic[0].S"])


def test_fisher_ignores_per_species_bounds(two_pool):
    """Bounds play no part in the Fisher information: a per-species dict gives
    the same matrix as the default."""
    model, reactor = two_pool
    ex = Experiment(jnp.asarray([0.3, 0.7, 0.0]), T_OBS, sigma=SIGMA, observed_species=["P"])
    a = aquakin.fisher_information(reactor, ex, ["fast.kS"], free_ic=FreeICConfig(["S", "X"]))
    b = aquakin.fisher_information(
        reactor, ex, ["fast.kS"], free_ic=FreeICConfig(["S", "X"], bounds={"S": (0.1, 0.4)})
    )
    assert a.names == b.names == ["fast.kS", "ic[0].S", "ic[0].X"]
    np.testing.assert_array_equal(a.matrix, b.matrix)


def test_fixed_total_with_rate_on_bound(two_pool):
    """A rate ending on its absolute bound next to a fixed-total split: the rate
    is flagged (the split's coordinates are unbounded, so never), its std is
    nan with zero covariance, the split keeps a finite std, the total is
    conserved, and fisher_information still equals the joint Hessian."""
    model, reactor = two_pool
    obs = _data(reactor, 0.3, 0.7)
    cfg = FreeICConfig(["S", "X"], fixed_total=True)
    with pytest.warns(RuntimeWarning, match="ended on an optimiser bound"):
        r = aquakin.calibrate(
            reactor,
            jnp.asarray([0.5, 0.5, 0.0]),
            obs,
            free_ic=cfg,
            laplace=LaplaceConfig(method="gauss_newton", include_ic=True),
            initial_params=model.default_parameters().at[model.param_index["fast.kS"]].set(0.3),
            **{
                **_COMMON,
                "optimizer": OptimizerConfig(
                    method="gauss_newton", bounds={"fast.kS": (None, 0.5)}
                ),
            },
        )
    assert r.at_bound == {"fast.kS": "upper"}
    assert r.params_named["fast.kS"] == pytest.approx(0.5, rel=1e-6)
    assert np.isnan(r.params_named_std["fast.kS"])
    assert np.all(np.asarray(r.joint_cov)[0] == 0.0)
    assert np.isfinite(r.ic_fractions_std[0]["S"]) and r.ic_fractions_std[0]["S"] > 0.0
    assert r.ic_named[0]["S"] + r.ic_named[0]["X"] == pytest.approx(1.0, rel=1e-12)
    fim = aquakin.fisher_information(
        reactor,
        Experiment(r.C0_fitted[0], T_OBS, sigma=SIGMA, observed_species=["P"]),
        ["fast.kS"],
        params=r.params,
        free_ic=cfg,
    )
    assert fim.names == r.joint_names
    np.testing.assert_allclose(fim.matrix, np.asarray(r.joint_hessian), rtol=1e-6)


def test_fixed_total_three_pools(two_pool):
    """Three pools (the measured product P included) share a total of 1.5: two
    log-ratio coordinates, the split recovered, and the fraction stds a
    delta-method projection of the joint covariance whose rows sum to zero."""
    model, reactor = two_pool
    C0_true = jnp.asarray([0.3, 0.7, 0.5])
    obs = reactor.solve(C0_true, t_span=(0.0, float(T_OBS[-1])), t_eval=T_OBS).C_named("P")
    r = aquakin.calibrate(
        reactor,
        jnp.asarray([0.5, 0.5, 0.5]),
        obs,
        free_ic=FreeICConfig(["S", "X", "P"], fixed_total=True),
        laplace=LaplaceConfig(method="gauss_newton", include_ic=True),
        **_COMMON,
    )
    assert r.joint_names == ["fast.kS", "ic[0].X:S", "ic[0].P:S"]
    np.testing.assert_allclose(
        [r.ic_named[0][s] for s in ("S", "X", "P")], [0.3, 0.7, 0.5], rtol=1e-3
    )
    assert sum(r.ic_named[0].values()) == pytest.approx(1.5, rel=1e-12)
    assert sum(r.ic_fractions[0].values()) == pytest.approx(1.0, rel=1e-12)
    from aquakin.integrate.calibrate import _simplex_fraction_cov

    z = [
        np.log(r.ic_named[0]["X"] / r.ic_named[0]["S"]),
        np.log(r.ic_named[0]["P"] / r.ic_named[0]["S"]),
    ]
    cov_f = _simplex_fraction_cov(np.asarray(z), np.asarray(r.joint_cov)[1:, 1:])
    np.testing.assert_allclose(
        [r.ic_fractions_std[0][s] for s in ("S", "X", "P")], np.sqrt(np.diag(cov_f)), rtol=1e-9
    )
    np.testing.assert_allclose(
        [r.ic_named_std[0][s] for s in ("S", "X", "P")], 1.5 * np.sqrt(np.diag(cov_f)), rtol=1e-9
    )


def test_fixed_total_single_pool_is_held(two_pool):
    """A one-pool fixed total has nothing to fit: the pool stays at its C0
    value with zero std, and only the rate enters the joint posterior."""
    model, reactor = two_pool
    obs = _data(reactor, 0.3, 0.7)
    r = aquakin.calibrate(
        reactor,
        jnp.asarray([0.3, 0.7, 0.0]),
        obs,
        free_ic=FreeICConfig(["S"], fixed_total=True),
        laplace=LaplaceConfig(method="gauss_newton", include_ic=True),
        **_COMMON,
    )
    assert r.joint_names == ["fast.kS"]
    assert r.ic_named[0]["S"] == pytest.approx(0.3, rel=1e-12)
    assert r.ic_fractions == [{"S": 1.0}]
    assert r.ic_fractions_std == [{"S": 0.0}]
    assert r.params_named["fast.kS"] == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize(
    "bounds", [{"S": (0.01, 1.0), "X": (0.05, 5.0)}, (0.01, 5.0)], ids=["dict", "pair"]
)
def test_profile_ic_with_per_species_bounds(two_pool, bounds):
    """Profiling a pool named in a per-species bounds dict drops its entry for
    the inner fits instead of failing every grid point; a shared pair passes
    through unchanged."""
    model, reactor = two_pool
    obs = _data(reactor, 0.3, 0.7)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        prof = aquakin.profile_likelihood(
            reactor,
            jnp.asarray([0.3, 0.7, 0.0]),
            obs,
            T_OBS,
            ["fast.kS"],
            grid=[0.25, 0.3, 0.35],
            profile_ic="S",
            observed_species=["P"],
            sigma=SIGMA,
            free_ic=FreeICConfig(["S", "X"], bounds=bounds),
            optimizer=OptimizerConfig(method="gauss_newton", n_starts=1),
        )
    assert np.all(np.isfinite(prof.loss))
    assert int(np.argmin(prof.loss)) == 1


def test_profile_ic_rejects_fixed_total_pool(two_pool):
    model, reactor = two_pool
    obs = _data(reactor, 0.3, 0.7)
    with pytest.raises(NotImplementedError, match="fixed_total"):
        aquakin.profile_likelihood(
            reactor,
            jnp.asarray([0.5, 0.5, 0.0]),
            obs,
            T_OBS,
            ["fast.kS"],
            grid=[0.3],
            profile_ic="S",
            observed_species=["P"],
            sigma=SIGMA,
            free_ic=FreeICConfig(["S", "X"], fixed_total=True),
        )


def test_profile_param_warm_start_with_fixed_total_split(two_pool):
    """Profiling a rate parameter with a fitted fixed-total split: the warm
    points start from the neighbour's pools (converted to the split coordinate)
    yet reach the same optimum as independent fits, and every point keeps the
    measured total."""
    model, reactor = two_pool
    obs = _data(reactor, 0.3, 0.7)
    kw = dict(
        grid=np.array([0.8, 1.0, 1.25]),
        profile_param="fast.kS",
        observed_species=["P"],
        loss="nll",
        sigma=SIGMA,
        free_ic=FreeICConfig(["S", "X"], fixed_total=True),
        optimizer=OptimizerConfig(method="gauss_newton", n_starts=1),
        polish=False,
    )
    args = (reactor, jnp.asarray([0.5, 0.5, 0.0]), obs, T_OBS, ["fast.kS", "slow.kX"])
    warm = aquakin.profile_likelihood(*args, warm_start=True, anchor=1.0, **kw)
    cold = aquakin.profile_likelihood(*args, warm_start=False, **kw)
    np.testing.assert_allclose(warm.loss, cold.loss, rtol=1e-6)
    assert int(np.argmin(warm.loss)) == 1
    for fit in warm.fits:
        assert fit.ic_named[0]["S"] + fit.ic_named[0]["X"] == pytest.approx(1.0, rel=1e-12)
