"""Integration tests for optimal experiment design.

For first-order decay ``A -> B`` observed through ``A(t) = A0 exp(-k t)`` with
only ``k`` estimated, one sample at time ``t`` carries information
``(t A0 exp(-k t) / sigma)^2`` about ``k``. Its D-criterion is maximised at
``t = 1/k``; averaged over draws of ``k`` (in log form) it is maximised at
``t = 1 / mean(k)``.
"""

import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import Experiment, FreeICConfig

K, A0, SIGMA = 0.25, 1.5, 0.01


@pytest.fixture(autouse=True)
def _release_compiled_programs():
    """Each design evaluation compiles a fresh forward-over-forward solve, and
    JAX keeps every compiled program for the life of the process. Free them
    after each test so a CI worker's memory does not grow test by test."""
    yield
    jax.clear_caches()
    gc.collect()


@pytest.fixture
def reactor(simple_model):
    return aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))


@pytest.fixture
def params(simple_model):
    return simple_model.default_parameters().at[0].set(K)


def _exp(times, a0=A0, **kw):
    return Experiment(
        jnp.asarray([a0, 0.0]),
        jnp.asarray(times, dtype=float),
        sigma=SIGMA,
        observed_species=["A"],
        **kw,
    )


def _info(t, a0=A0, k=K):
    """Information about k from one sample at t."""
    return (t * a0 * np.exp(-k * t) / SIGMA) ** 2


# ---------- criteria ----------


def test_design_criterion_values():
    F = np.array([[4.0, 1.0], [1.0, 2.0]])
    w = np.linalg.eigvalsh(F)
    crit = aquakin.design_criterion
    assert crit(F, "D") == pytest.approx(np.log(np.linalg.det(F)))
    assert crit(F, "A") == pytest.approx(np.trace(np.linalg.inv(F)))
    assert crit(F, "E") == pytest.approx(w[0])
    assert crit(F, "modified_E") == pytest.approx(w[1] / w[0])
    # Subset: the Schur complement 4 - 1^2 / 2.
    assert crit(F, "D", subset=["a"], names=["a", "b"]) == pytest.approx(np.log(3.5))
    assert crit(F, "D", subset=["a", "b"], names=["a", "b"]) == pytest.approx(crit(F, "D"))

    S = np.ones((2, 2))
    assert crit(S, "D") == -np.inf
    assert crit(S, "A") == np.inf
    assert crit(S, "modified_E") == np.inf
    assert crit(S, "E") == pytest.approx(0.0, abs=1e-12)


def test_round_off_eigenvalue_counts_as_singular():
    # A null direction computed in floating point has an eigenvalue of
    # round-off size and either sign. Both signs must score as singular, as
    # identifiability reports them, rather than a finite D for the positive one.
    crit = aquakin.design_criterion
    for tiny in (1e-17, -1e-17):
        F = np.diag([1.0, tiny])
        assert crit(F, "D") == -np.inf
        assert crit(F, "A") == np.inf
        assert crit(F, "modified_E") == np.inf
        assert crit(F, "E") == 0.0
    # A small but genuine eigenvalue is still scored normally.
    F = np.diag([1.0, 1e-8])
    assert crit(F, "D") == pytest.approx(np.log(1e-8))
    assert crit(F, "E") == pytest.approx(1e-8)


def test_design_criterion_from_fisher_information(reactor, params):
    fim = aquakin.fisher_information(
        reactor, _exp([2.0, 6.0]), ["A_to_B.k"], params=params, free_ic=FreeICConfig(["A"])
    )
    assert aquakin.design_criterion(fim) == pytest.approx(np.log(np.linalg.det(fim.matrix)))
    F = fim.matrix
    schur = F[0, 0] - F[0, 1] ** 2 / F[1, 1]
    assert aquakin.design_criterion(fim, subset=["A_to_B.k"]) == pytest.approx(np.log(schur))


def test_design_criterion_validation():
    with pytest.raises(ValueError, match="Unknown criterion"):
        aquakin.design_criterion(np.eye(2), "Z")
    with pytest.raises(ValueError, match="names="):
        aquakin.design_criterion(np.eye(2), subset=["a"])
    with pytest.raises(KeyError, match="Unknown subset name"):
        aquakin.design_criterion(np.eye(2), subset=["z"], names=["a", "b"])


# ---------- candidate ranking ----------


def test_rank_experiments_orders_by_information(reactor, params):
    cands = [_exp([1.0]), _exp([4.0]), _exp([10.0])]
    ranked = aquakin.rank_experiments(reactor, cands, ["A_to_B.k"], params=params)
    assert [c.index for c in ranked] == [1, 2, 0]
    assert ranked[0].value == pytest.approx(np.log(_info(4.0)), rel=1e-6)
    assert all(c.improvement is None for c in ranked)

    # With a base, the improvement is the gain in log det.
    ranked = aquakin.rank_experiments(
        reactor, cands, ["A_to_B.k"], base=[_exp([2.0])], params=params
    )
    gain = np.log((_info(2.0) + _info(4.0)) / _info(2.0))
    assert ranked[0].index == 1
    assert ranked[0].improvement == pytest.approx(gain, rel=1e-6)

    # A is smaller-is-better: the best candidate has the smallest value and a
    # positive improvement.
    ranked = aquakin.rank_experiments(
        reactor, cands, ["A_to_B.k"], base=[_exp([2.0])], params=params, criterion="A"
    )
    assert ranked[0].index == 1
    assert ranked[0].value == pytest.approx(1.0 / (_info(2.0) + _info(4.0)), rel=1e-5)
    assert ranked[0].improvement > 0.0
    assert ranked[0].value < ranked[-1].value


def test_rank_experiments_draws_and_subset(reactor, params, simple_model):
    draws = jnp.stack([params.at[0].set(0.2), params.at[0].set(0.3)])
    ranked = aquakin.rank_experiments(
        reactor, [_exp([4.0]), _exp([1.0])], ["A_to_B.k"], param_draws=draws
    )
    expected = np.mean([np.log(_info(4.0, k=k)) for k in (0.2, 0.3)])
    assert ranked[0].index == 0
    assert ranked[0].value == pytest.approx(expected, rel=1e-6)

    # Ds on the rate with the candidate's pool as nuisance.
    ranked = aquakin.rank_experiments(
        reactor,
        [_exp([2.0, 6.0])],
        ["A_to_B.k"],
        params=params,
        free_ic=FreeICConfig(["A"]),
        subset=["A_to_B.k"],
    )
    fim = aquakin.fisher_information(
        reactor, _exp([2.0, 6.0]), ["A_to_B.k"], params=params, free_ic=FreeICConfig(["A"])
    )
    assert ranked[0].value == pytest.approx(
        aquakin.design_criterion(fim, subset=["A_to_B.k"]), rel=1e-6
    )

    with pytest.raises(ValueError, match="at least one"):
        aquakin.rank_experiments(reactor, [], ["A_to_B.k"])
    with pytest.raises(ValueError, match="param_draws has"):
        aquakin.rank_experiments(reactor, [_exp([1.0])], ["A_to_B.k"], param_draws=jnp.ones(3))


# ---------- gradient-based optimisation ----------


def test_d_optimal_single_time_is_one_over_k(reactor, params):
    res = aquakin.optimize_experiment(
        reactor, _exp([1.0]), ["A_to_B.k"], params=params, time_bounds=(0.1, 20.0)
    )
    assert res.success
    assert float(res.experiments[0].t_obs[0]) == pytest.approx(1.0 / K, rel=1e-4)
    assert res.value == pytest.approx(np.log(_info(1.0 / K)), rel=1e-6)
    assert res.initial_value == pytest.approx(np.log(_info(1.0)), rel=1e-6)


def test_robust_design_over_draws(reactor, params):
    draws = jnp.stack([params.at[0].set(0.2), params.at[0].set(0.3)])
    res = aquakin.optimize_experiment(
        reactor, _exp([1.0]), ["A_to_B.k"], param_draws=draws, time_bounds=(0.1, 20.0)
    )
    assert float(res.experiments[0].t_obs[0]) == pytest.approx(1.0 / 0.25, rel=1e-4)


def test_min_spacing_is_respected(reactor, params):
    res = aquakin.optimize_experiment(
        reactor,
        _exp([1.0, 8.0]),
        ["A_to_B.k"],
        params=params,
        time_bounds=(0.1, 20.0),
        min_spacing=2.0,
    )
    t = np.asarray(res.experiments[0].t_obs)
    assert t[1] - t[0] == pytest.approx(2.0, abs=1e-5)  # both want t = 4: the gap binds
    assert t[0] < 1.0 / K < t[1]
    assert res.value > res.initial_value


def test_initial_concentration_goes_to_its_bound(reactor, params):
    """Information about k grows with A0^2, so A0 moves to its upper bound."""
    res = aquakin.optimize_experiment(
        reactor,
        _exp([4.0]),
        ["A_to_B.k"],
        params=params,
        optimize_times=False,
        initial_bounds={"A": (0.5, 3.0)},
    )
    assert float(res.experiments[0].C0[0]) == pytest.approx(3.0)
    np.testing.assert_allclose(res.experiments[0].t_obs, [4.0])
    assert res.value - res.initial_value == pytest.approx(2.0 * np.log(3.0 / A0), rel=1e-6)

    keyed = aquakin.optimize_experiment(
        reactor,
        [_exp([4.0]), _exp([4.0])],
        ["A_to_B.k"],
        params=params,
        optimize_times=False,
        initial_bounds={(1, "A"): (0.5, 3.0)},
    )
    assert float(keyed.experiments[0].C0[0]) == pytest.approx(A0)
    assert float(keyed.experiments[1].C0[0]) == pytest.approx(3.0)


def test_base_and_time_unit(reactor, params):
    """With a fixed base the new sample still goes to t = 1/k; in minutes that
    is (1/k)/60."""
    res = aquakin.optimize_experiment(
        reactor,
        _exp([1.0 / 60.0], time_unit="min"),
        ["A_to_B.k"],
        params=params,
        base=[_exp([2.0])],
        time_bounds=(0.001, 0.3),
    )
    assert float(res.experiments[0].t_obs[0]) == pytest.approx(1.0 / K / 60.0, rel=1e-4)
    assert res.value == pytest.approx(np.log(_info(2.0) + _info(1.0 / K)), rel=1e-6)


@pytest.mark.parametrize("criterion", ["A", "E", "modified_E"])
def test_other_criteria_improve(reactor, params, criterion):
    res = aquakin.optimize_experiment(
        reactor,
        _exp([1.0, 2.0, 3.0]),
        ["A_to_B.k"],
        params=params,
        free_ic=FreeICConfig(["A"]),
        criterion=criterion,
        time_bounds=(0.1, 20.0),
        min_spacing=0.5,
    )
    better = res.value > res.initial_value if criterion == "E" else res.value < res.initial_value
    assert better


def test_optimize_validation(reactor, params):
    k = ["A_to_B.k"]
    opt = aquakin.optimize_experiment
    with pytest.raises(ValueError, match="Nothing to optimise"):
        opt(reactor, _exp([1.0]), k, optimize_times=False)
    with pytest.raises(ValueError, match="at least one"):
        opt(reactor, [], k)
    with pytest.raises(ValueError, match="outside time_bounds"):
        opt(reactor, _exp([5.0]), k, time_bounds=(0.1, 2.0))
    with pytest.raises(ValueError, match="minimum spacing"):
        opt(reactor, _exp([1.0, 1.5]), k, min_spacing=1.0, time_bounds=(0.1, 5.0))
    with pytest.raises(ValueError, match="lo < hi"):
        opt(reactor, _exp([1.0]), k, time_bounds=(3.0, 1.0))
    with pytest.raises(ValueError, match="share one time_unit"):
        opt(reactor, [_exp([1.0]), _exp([1.0], time_unit="min")], k)
    with pytest.raises(ValueError, match="outside"):
        opt(reactor, _exp([1.0]), k, optimize_times=False, initial_bounds={"A": (2.0, 3.0)})
    with pytest.raises(KeyError, match="Unknown species"):
        opt(reactor, _exp([1.0]), k, initial_bounds={"Z": (1.0, 2.0)})
    with pytest.raises(ValueError, match="refers to experiment"):
        opt(reactor, _exp([1.0]), k, initial_bounds={(3, "A"): (1.0, 2.0)})
    with pytest.raises(ValueError, match="lo < hi"):
        opt(reactor, _exp([1.0]), k, initial_bounds={"A": (2.0, 1.0)})
    # One sample cannot separate k from A0: singular for every criterion.
    with pytest.raises(ValueError, match="singular"):
        opt(reactor, _exp([1.0]), k, params=params, free_ic=FreeICConfig(["A"]))
    with pytest.raises(ValueError, match="singular"):
        opt(reactor, _exp([1.0]), k, params=params, free_ic=FreeICConfig(["A"]), criterion="E")
    with pytest.raises(ValueError, match="Unknown criterion"):
        opt(reactor, _exp([1.0]), k, criterion="Z")


def test_nothing_to_estimate(reactor):
    with pytest.raises(ValueError, match="Nothing to estimate"):
        aquakin.fisher_information(reactor, _exp([1.0]), [])
