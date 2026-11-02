"""Unit tests for :func:`aquakin.design_from_candidates`.

The relaxed method is checked against the classical D-optimal designs for
polynomial regression on [-1, 1]; the integer methods against brute force on
small pools; and the constraints, subset scoring, regularization and robust
(parameter-draw) design directly.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin import design_criterion, design_from_candidates

X = np.linspace(-1.0, 1.0, 21)


def _poly(degree):
    """Rank-one information of one observation of a polynomial at each x."""
    f = np.stack([X**k for k in range(degree + 1)], axis=1)
    return np.einsum("ci,cj->cij", f, f)


def _random_pool(seed, n_c=10, p=3, rank=2):
    g = np.random.default_rng(seed).normal(size=(n_c, rank, p))
    return np.einsum("cki,ckj->cij", g, g)


def _brute_force(F, k, base, criterion="D"):
    sign = 1.0 if criterion in ("D", "E") else -1.0
    return max(
        sign * design_criterion(base + F[list(c)].sum(axis=0), criterion)
        for c in itertools.combinations(range(len(F)), k)
    )


def test_relaxed_straight_line_puts_half_at_each_end():
    r = design_from_candidates(_poly(1), budget=1.0, max_repeats=None, method="relaxed")
    expected = np.zeros(len(X))
    expected[[0, -1]] = 0.5
    np.testing.assert_allclose(r.relaxed_weights, expected, atol=1e-6)
    max_var, q = r.equivalence
    assert q == 2.0
    assert max_var == pytest.approx(2.0, rel=1e-6)
    assert r.gap < 1e-8
    # The relaxed optimum's information is diag(1, 1): log det 0.
    assert r.relaxed_value == pytest.approx(0.0, abs=1e-8)


def test_relaxed_quadratic_puts_a_third_at_ends_and_centre():
    r = design_from_candidates(_poly(2), budget=1.0, max_repeats=None, method="relaxed")
    expected = np.zeros(len(X))
    expected[[0, 10, -1]] = 1.0 / 3.0
    np.testing.assert_allclose(r.relaxed_weights, expected, atol=1e-5)
    assert r.equivalence[0] == pytest.approx(3.0, rel=1e-5)
    assert r.equivalence[1] == 3.0


def test_integer_design_replicates_the_relaxed_support():
    """With replicates allowed, 4 straight-line samples go 2 + 2 to the ends,
    and the relaxed optimum is then attained (efficiency 1)."""
    r = design_from_candidates(_poly(1), budget=4, max_repeats=None)
    assert r.counts[0] == 2 and r.counts[-1] == 2 and r.counts.sum() == 4
    assert r.selected == [0, 20]
    assert r.value == pytest.approx(np.log(16.0))
    assert r.bound == pytest.approx(np.log(16.0), abs=1e-7)
    assert r.efficiency == pytest.approx(1.0, abs=1e-7)


@pytest.mark.parametrize("seed", [0, 1, 4, 5])
@pytest.mark.parametrize("criterion", ["D", "A"])
def test_exchange_and_auto_match_brute_force(seed, criterion):
    F, base = _random_pool(seed), 0.1 * np.eye(3)
    best = _brute_force(F, 3, base, criterion)
    sign = 1.0 if criterion == "D" else -1.0
    for method in ("exchange", "auto"):
        r = design_from_candidates(F, budget=3, base=base, criterion=criterion, method=method)
        assert sign * r.value == pytest.approx(best, rel=1e-10)
        assert r.counts.sum() == 3 and r.counts.max() == 1
    # The relaxed bound holds for every integer design.
    r = design_from_candidates(F, budget=3, base=base, criterion=criterion)
    assert sign * r.bound >= best - 1e-9
    assert 0.0 < r.efficiency <= 1.0


def test_auto_is_exact_on_small_pools():
    """Exhaustive scoring makes auto exact where exchange stops at a local
    optimum (this pool has one for D)."""
    F, base = _random_pool(16), 0.1 * np.eye(3)
    best = _brute_force(F, 3, base)
    ex = design_from_candidates(F, budget=3, base=base, method="exchange")
    auto = design_from_candidates(F, budget=3, base=base)
    assert ex.value < best - 1e-6
    assert auto.value == pytest.approx(best, rel=1e-12)
    assert auto.method == "exhaustive"
    # Without enumeration, auto still does at least as well as exchange.
    capped = design_from_candidates(F, budget=3, base=base, exhaustive_limit=0)
    assert capped.method != "exhaustive"
    assert capped.value >= ex.value - 1e-12


@pytest.mark.parametrize("seed", range(6))
def test_greedy_is_within_one_minus_one_over_e(seed):
    """With a positive-definite base, log det gain is monotone submodular, so
    greedy gets at least (1 - 1/e) of the best subset's gain."""
    F, base = _random_pool(seed), 0.1 * np.eye(3)
    best = _brute_force(F, 3, base)
    base_value = design_criterion(base, "D")
    g = design_from_candidates(F, budget=3, base=base, method="greedy")
    assert g.method == "greedy"
    assert g.value - base_value >= (1.0 - np.exp(-1.0)) * (best - base_value) - 1e-12
    ex = design_from_candidates(F, budget=3, base=base, method="exchange")
    assert ex.value >= g.value - 1e-12


def test_ties_break_by_index_and_results_are_deterministic():
    F = np.stack([np.eye(2)] * 5)
    r = design_from_candidates(F, budget=2, method="greedy")
    assert r.selected == [0, 1]
    again = design_from_candidates(F, budget=2, method="greedy")
    np.testing.assert_array_equal(r.counts, again.counts)


def test_costs_required_and_max_repeats_are_honoured():
    F = _poly(1)
    costs = np.ones(len(X))
    costs[0] = 3.0  # the left end is expensive
    r = design_from_candidates(F, budget=4, costs=costs, max_repeats=2, method="exchange")
    assert r.cost <= 4.0 + 1e-12
    assert r.counts.max() <= 2
    # Required candidates are always in the design, even uninformative ones.
    r = design_from_candidates(F, budget=3, required=[10])
    assert r.counts[10] >= 1 and r.counts.sum() == 3
    # Per-candidate caps.
    caps = np.ones(len(X), dtype=int)
    caps[-1] = 3
    r = design_from_candidates(F, budget=6, max_repeats=caps)
    assert r.counts[-1] <= 3 and np.all(r.counts[:-1] <= 1)
    assert r.counts.sum() == 6


def test_infeasible_budgets_raise():
    F = _poly(1)
    with pytest.raises(ValueError, match="required candidates cost"):
        design_from_candidates(F, budget=1, required=[0, 1])
    with pytest.raises(ValueError, match="buys no candidate"):
        design_from_candidates(F, budget=0.5)
    with pytest.raises(ValueError, match="positive"):
        design_from_candidates(F, budget=0.0)
    with pytest.raises(ValueError, match="costs"):
        design_from_candidates(F, budget=2, costs=np.zeros(len(X)))
    with pytest.raises(ValueError, match="max_repeats"):
        design_from_candidates(F, budget=2, max_repeats=0)
    with pytest.raises(KeyError, match="required"):
        design_from_candidates(F, budget=2, required=[99])


def test_subset_matches_schur_complement_of_design_criterion():
    F = _poly(2)
    names = ["a", "b", "c"]
    r = design_from_candidates(F, budget=5, names=names, subset=["b", "c"], max_repeats=None)
    M = np.einsum("c,cij->ij", r.counts, F)
    assert r.value == pytest.approx(design_criterion(M, "D", subset=["b", "c"], names=names))
    np.testing.assert_allclose(r.matrix, M)
    assert r.equivalence[1] == 2.0
    assert r.equivalence[0] == pytest.approx(2.0, rel=1e-5)
    with pytest.raises(ValueError, match="names"):
        design_from_candidates(F, budget=5, subset=["b"])


def test_singular_base_is_regularized_and_singular_pool_raises():
    """No base: early designs are singular, so selection uses a small ridge
    while the reported value is the plain criterion. A pool that cannot
    determine every parameter raises with guidance."""
    F = _random_pool(0, rank=1)
    best = _brute_force(F, 4, np.zeros((3, 3)))
    r = design_from_candidates(F, budget=4)
    assert r.regularized
    assert r.value == pytest.approx(best, rel=1e-6)
    for method in ("greedy", "exchange"):
        r = design_from_candidates(F, budget=4, method=method)
        assert np.isfinite(r.value)
    pos = design_from_candidates(F, budget=4, base=np.eye(3))
    assert not pos.regularized

    flat = F.copy()
    flat[:, 2, :] = flat[:, :, 2] = 0.0  # nothing informs the third parameter
    with pytest.raises(ValueError, match="cannot determine"):
        design_from_candidates(flat, budget=4)
    # ... but a prior on it is enough.
    r = design_from_candidates(flat, budget=4, base=np.diag([0.0, 0.0, 1.0]))
    assert np.isfinite(r.value)


def test_param_draws_average_the_criterion():
    """Identical draws reproduce the single design; the criterion is the mean
    over draws; the draw axis can be first or second."""
    F = _random_pool(2)
    base = 0.1 * np.eye(3)
    single = design_from_candidates(F, budget=3, base=base)
    stacked = design_from_candidates(np.stack([F, F]), budget=3, base=base, param_draws_axis=0)
    np.testing.assert_array_equal(stacked.counts, single.counts)
    assert stacked.value == pytest.approx(single.value)

    G = _random_pool(3)
    draws = np.stack([F, G])
    r = design_from_candidates(draws, budget=3, base=base, param_draws_axis=0)
    chosen = r.counts.astype(bool)
    mean = 0.5 * (
        design_criterion(base + F[chosen].sum(0), "D")
        + design_criterion(base + G[chosen].sum(0), "D")
    )
    assert r.value == pytest.approx(mean)
    assert r.matrix.shape == (3, 3)
    swapped = design_from_candidates(
        np.swapaxes(draws, 0, 1), budget=3, base=base, param_draws_axis=1
    )
    np.testing.assert_array_equal(swapped.counts, r.counts)


def test_gains_and_non_relaxable_criteria():
    F, base = _random_pool(1), 0.1 * np.eye(3)
    r = design_from_candidates(F, budget=3, base=base)
    after = [design_criterion(r.matrix + F[c], "D") - r.value for c in range(len(F))]
    np.testing.assert_allclose(r.gains, after, rtol=1e-10)
    for crit in ("E", "modified_E"):
        with pytest.raises(ValueError, match="relaxed"):
            design_from_candidates(F, budget=3, base=base, criterion=crit, method="relaxed")
        e = design_from_candidates(F, budget=3, base=base, criterion=crit)
        assert e.relaxed_weights is None and e.bound is None
    best_e = _brute_force(F, 3, base, "E")
    assert design_from_candidates(F, budget=3, base=base, criterion="E").value == pytest.approx(
        best_e
    )


def test_pool_from_fisher_information_from_fn_groups():
    """The groups returned by fisher_information_from_fn are a labelled pool;
    required candidates are given by label."""
    t = jnp.linspace(0.0, 4.0, 9)

    def fn(theta):
        a, k = theta
        return a * jnp.exp(-k * t)

    _, per_time = aquakin.fisher_information_from_fn(
        fn,
        jnp.array([2.0, 0.7]),
        names=["a", "k"],
        sigma=0.05,
        groups=[f"t{i}" for i in range(9)],
    )
    r = design_from_candidates(per_time, budget=3, required=["t8"])
    assert r.labels == [f"t{i}" for i in range(9)]
    assert "t8" in r.selected and len(r.selected) == 3
    # For exponential decay the information about k peaks at t = 1/k, so the
    # choice must include an early sample (for a) and one near 1/k.
    assert "t0" in r.selected
    with pytest.raises(KeyError):
        design_from_candidates(per_time, budget=3, required=["t9"])


def test_pool_and_option_validation():
    F = _poly(1)
    info = [
        aquakin.FisherInformation(
            matrix=m,
            names=["a", "b"],
            transforms=["none"] * 2,
            values={},
            jacobian=np.zeros((0, 2)),
        )
        for m in F
    ]
    # A list of FisherInformation is a pool labelled by index.
    r = design_from_candidates(info, budget=2)
    assert r.labels == list(range(len(X)))
    assert r.selected == [0, 20]
    with pytest.raises(ValueError, match="Unknown method"):
        design_from_candidates(F, budget=2, method="random")
    with pytest.raises(ValueError, match="FisherInformation objects"):
        design_from_candidates({"a": F[0]}, budget=1)
    with pytest.raises(ValueError, match="FisherInformation objects"):
        design_from_candidates({}, budget=1)
    renamed = info[:2] + [
        aquakin.FisherInformation(
            matrix=F[2],
            names=["a", "c"],
            transforms=["none"] * 2,
            values={},
            jacobian=np.zeros((0, 2)),
        )
    ]
    with pytest.raises(ValueError, match="same names"):
        design_from_candidates(renamed, budget=2)
    with pytest.raises(ValueError, match="disagrees"):
        design_from_candidates(info, budget=2, names=["x", "y"])
    with pytest.raises(ValueError, match="only to an array"):
        design_from_candidates(info, budget=2, param_draws_axis=0)
    with pytest.raises(ValueError, match="param_draws_axis="):
        design_from_candidates(np.stack([F, F]), budget=2)
    with pytest.raises(ValueError, match="4-D"):
        design_from_candidates(F, budget=2, param_draws_axis=0)
    with pytest.raises(ValueError, match="4-D"):
        design_from_candidates(np.stack([F, F]), budget=2, param_draws_axis=2)
    with pytest.raises(ValueError, match="square"):
        design_from_candidates(np.zeros((3, 2, 3)), budget=2)
    with pytest.raises(ValueError, match="names has"):
        design_from_candidates(F, budget=2, names=["a"])
    with pytest.raises(ValueError, match="base must be"):
        design_from_candidates(F, budget=2, base=np.eye(3))
    # A FisherInformation base, and a per-draw base.
    r = design_from_candidates(info, budget=2, base=info[10])
    assert r.regularized  # one rank-one sample is still singular
    assert np.isfinite(r.value)
    r = design_from_candidates(
        np.stack([F, F]), budget=2, base=np.stack([np.eye(2), 2 * np.eye(2)]), param_draws_axis=0
    )
    assert np.isfinite(r.value)


def test_relaxed_a_criterion_on_a_subset_and_iteration_cap():
    """A on a subset gives a lower bound on the trace; a solver stopped after
    one iteration still returns a feasible relaxed design and an honest gap."""
    F = _poly(2)
    names = ["a", "b", "c"]
    r = design_from_candidates(F, budget=3, names=names, subset=["b"], criterion="A")
    assert r.bound <= r.value + 1e-9
    assert 0.0 < r.efficiency <= 1.0
    assert r.equivalence is None
    capped = design_from_candidates(F, budget=3, method="relaxed", max_iter=1)
    assert capped.relaxed_weights.sum() == pytest.approx(3.0)
    assert capped.gap > 1e-6
    full = design_from_candidates(F, budget=3, method="relaxed")
    assert full.gap < capped.gap


def test_modified_e_greedy_stops_and_exchange_adds():
    """modified-E can get worse with more information, so once the design is
    non-singular greedy stops when nothing improves it. Before that it always
    adds, even though the ridge-only start already has condition number 1."""
    F = np.stack([np.diag([1.0, 1.0]), np.diag([4.0, 0.0]), np.diag([0.0, 0.5])])
    g = design_from_candidates(F, budget=3, criterion="modified_E", method="greedy")
    assert g.counts.tolist() == [1, 0, 0]
    assert g.value == pytest.approx(1.0)
    ex = design_from_candidates(F, budget=3, criterion="modified_E", method="exchange")
    assert ex.value <= g.value


def test_exchange_skips_removals_with_nothing_to_swap_in():
    """When every other candidate is at its cap, dropping one leaves nothing
    to add in its place, and exchange keeps the design."""
    F = np.stack([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])])
    r = design_from_candidates(F, budget=2, method="exchange", base=0.1 * np.eye(2))
    assert r.counts.tolist() == [1, 1]


def test_internal_singular_and_projection_edges():
    """Edge cases of the relaxed solver's internals that a well-posed pool
    reaches only in a line search."""
    from aquakin.integrate.candidate_design import (
        _Context,
        _exchange,
        _greedy,
        _project,
        _solve_relaxed,
        _value_and_grad,
    )

    F = _poly(2)[None]
    zero = np.zeros((1, 3, 3))
    w = np.zeros(len(X))
    w[0] = 1.0  # a single rank-one sample: singular
    for crit, sub, nui in [
        ("D", None, None),
        ("D", np.array([0]), np.array([1, 2])),
        ("A", None, None),
    ]:
        f, g = _value_and_grad(w, F, zero, crit, sub, nui)
        assert f == -np.inf and g is None
    # An invertible but indefinite matrix (as a line search might reach).
    w[[10, 20]] = 1.0
    neg = zero.copy()
    neg[0, 0, 0] = -10.0
    assert _value_and_grad(w, F, neg, "D", None, None)[1] is None
    assert _value_and_grad(w, F, neg, "A", None, None)[1] is None
    neg_nn = zero.copy()
    neg_nn[0, 1, 1] = -10.0
    assert _value_and_grad(w, F, neg_nn, "D", np.array([0]), np.array([1, 2]))[1] is None
    nan = np.full((1, 3, 3), np.nan)
    with np.errstate(invalid="ignore"):
        assert _value_and_grad(w, F, nan, "D", None, None)[1] is None

    # Projection onto an uncapped budget from far below it.
    cost = np.ones(4)
    p = _project(np.array([-50.0, -60.0, -70.0, -80.0]), cost, 2.0, np.zeros(4), np.full(4, np.inf))
    assert p @ cost == pytest.approx(2.0) and np.all(p >= 0.0)

    lower, upper = np.zeros(len(X)), np.ones(len(X))
    with pytest.raises(ValueError, match="relaxed design is singular"):
        _solve_relaxed(
            np.zeros_like(F), zero, np.ones(len(X)), 2.0, lower, upper, "D", None, None, 1e-9, 10
        )

    ctx = _Context(3, np.ones(3), 2.0, np.zeros(3), np.ones(3), "D")
    # Greedy stops when no addition has a finite value.
    n = _greedy(np.zeros(3), ctx, lambda n: 0.0, lambda n: np.full(3, -np.inf))
    assert n.tolist() == [0.0, 0.0, 0.0]
    # Exchange stops at its pass limit even while it is still improving.
    n = _exchange(
        np.zeros(3),
        ctx,
        lambda n: float(n.sum()),
        lambda n: n.sum() + 1.0 + np.zeros(3),
        max_passes=1,
    )
    assert n.sum() == 1.0
