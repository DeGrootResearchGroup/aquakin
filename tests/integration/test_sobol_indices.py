"""Variance-based global sensitivity: ``aquakin.sobol_indices``.

Values are checked against closed forms: the Ishigami function (first-order,
total and interaction indices known exactly), a linear additive model under
normal inputs, a pure-interaction product, and first-order decay through
``reactor.solve``. The bootstrap intervals are checked for coverage over
independent seeds, and the total indices against the DGSM upper bound.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin

PI = np.pi
ISHIGAMI_A, ISHIGAMI_B = 7.0, 0.1


def _ishigami(x):
    return jnp.sin(x[0]) + ISHIGAMI_A * jnp.sin(x[1]) ** 2 + ISHIGAMI_B * x[2] ** 4 * jnp.sin(x[0])


def _ishigami_indices():
    """Closed-form (S, S_tot) of the Ishigami function on [-pi, pi]^3."""
    a, b = ISHIGAMI_A, ISHIGAMI_B
    V1 = 0.5 * (1.0 + b * PI**4 / 5.0) ** 2
    V2 = a**2 / 8.0
    V13 = 8.0 * b**2 * PI**8 / 225.0
    V = V1 + V2 + V13
    return np.array([V1, V2, 0.0]) / V, np.array([V1 + V13, V2, V13]) / V


ISHIGAMI_RANGES = [(-PI, PI)] * 3


def test_ishigami_first_and_total_indices_match_closed_form():
    """S_j and S_j^tot of the Ishigami function, and its x1-x3 interaction.

    Tolerance: abs 0.01 is about the iid standard error at N = 4096 (the
    bootstrap half-widths here are 0.013 to 0.04); scrambled Sobol beats it by
    an order of magnitude (errors <= 2.2e-3 over seeds 0-4).
    """
    S, ST = _ishigami_indices()
    res = aquakin.sobol_indices(_ishigami, ISHIGAMI_RANGES, n_samples=4096, seed=0)
    np.testing.assert_allclose(res.first_order, S, rtol=0, atol=0.01)
    np.testing.assert_allclose(res.total_order, ST, rtol=0, atol=0.01)
    # x2 enters additively (no interaction); x1 and x3 interact only together.
    np.testing.assert_allclose(res.interaction, ST - S, rtol=0, atol=0.01)
    assert res.interaction[2] == pytest.approx(ST[2], abs=0.01)
    assert res.n_samples == 4096
    assert res.n_evaluations == 4096 * 5 == aquakin.sobol_indices_cost(3, 4096)
    assert res.n_valid == 4096
    assert [n for n, _ in res.ranked()] == ["z0", "z1", "z2"]
    for ci, exact in ((res.first_order_ci, S), (res.total_order_ci, ST)):
        assert np.all((ci[:, 0] <= exact) & (exact <= ci[:, 1]))


def test_bootstrap_intervals_cover_the_ishigami_indices():
    """Over 40 independent seeds at N = 256, each index's 95 % interval covers
    the closed-form value at least 85 % of the time.

    The bootstrap treats the base sample as independent draws, which makes it
    conservative on a scrambled-Sobol design (measured coverage 0.95-1.0 here);
    85 % leaves three binomial standard errors (0.034 at 40 draws) below the
    nominal level.
    """
    S, ST = _ishigami_indices()
    hits = []
    for seed in range(40):
        res = aquakin.sobol_indices(_ishigami, ISHIGAMI_RANGES, n_samples=256, seed=seed)
        lo = np.concatenate([res.first_order_ci[:, 0], res.total_order_ci[:, 0]])
        hi = np.concatenate([res.first_order_ci[:, 1], res.total_order_ci[:, 1]])
        exact = np.concatenate([S, ST])
        hits.append((lo <= exact) & (exact <= hi))
    coverage = np.mean(hits, axis=0)
    assert np.all(coverage >= 0.85), coverage


def test_linear_model_under_normal_inputs_matches_closed_form():
    """f = sum a_j z_j with z_j ~ N(mu_j, sigma_j^2): S_j = S_j^tot =
    a_j^2 sigma_j^2 / sum_k a_k^2 sigma_k^2, and no interaction.

    Tolerance: abs 0.01, as for Ishigami at the same N (an additive model is
    easier for both estimators).
    """
    a = np.array([1.0, -2.0, 0.5])
    sigma = np.array([1.0, 0.5, 2.0])
    share = a**2 * sigma**2 / np.sum(a**2 * sigma**2)
    dists = {
        "u": {"dist": "normal", "mean": 3.0, "std": 1.0},
        "v": {"dist": "normal", "mean": -1.0, "std": 0.5},
        "w": {"dist": "normal", "mean": 10.0, "std": 2.0},
    }
    res = aquakin.sobol_indices(lambda z: jnp.dot(jnp.asarray(a), z), dists, n_samples=4096)
    assert res.input_names == ["u", "v", "w"]
    np.testing.assert_allclose(res.first_order, share, rtol=0, atol=0.01)
    np.testing.assert_allclose(res.total_order, share, rtol=0, atol=0.01)
    np.testing.assert_allclose(res.interaction, 0.0, rtol=0, atol=0.01)


def test_pure_interaction_product_has_zero_first_order_and_unit_total():
    """f = z0 z1 with z uniform on [-1, 1]: E[f | z_j] = 0, so S_j = 0, and all
    the variance is the interaction, so S_j^tot = 1. Tolerance as for Ishigami."""
    res = aquakin.sobol_indices(lambda z: z[0] * z[1], [(-1.0, 1.0)] * 2, n_samples=4096)
    np.testing.assert_allclose(res.first_order, 0.0, rtol=0, atol=0.01)
    np.testing.assert_allclose(res.total_order, 1.0, rtol=0, atol=0.01)
    np.testing.assert_allclose(res.interaction, 1.0, rtol=0, atol=0.01)
    assert res.output_variance == pytest.approx(1.0 / 9.0, rel=0.01)


def test_total_index_is_below_the_dgsm_bound():
    """On the shared Ishigami problem every S_j^tot lies below the DGSM upper
    bound nu_j (b_j - a_j)^2 / (pi^2 Var f) (Lamboni, Sobol & Kucherenko 2013).

    The closest pair is x3 (S^tot 0.244 against a bound of about 0.48), so the
    margin is far wider than either estimate's error at these sample sizes.
    """
    sob = aquakin.sobol_indices(_ishigami, ISHIGAMI_RANGES, n_samples=4096)
    dg = aquakin.dgsm(_ishigami, ISHIGAMI_RANGES, n_samples=1024)
    bound = np.asarray(dg.sobol_total_bound)
    assert np.all(sob.total_order <= bound), (sob.total_order, bound)
    assert np.all(sob.total_order_ci[:, 1] <= bound)


def _decay_indices(t, k_range, a_range):
    """Closed-form (S, S_tot) of B(t) = A0 (1 - exp(-k t)) for k and A0 uniform
    and independent: the output is a product g(k) A0, so with m_g, v_g and m_a,
    v_a the means and variances, V_k = m_a^2 v_g, V_A0 = m_g^2 v_a and the
    interaction V_k,A0 = v_g v_a."""
    (k1, k2), (a1, a2) = k_range, a_range
    e1 = (np.exp(-k1 * t) - np.exp(-k2 * t)) / (t * (k2 - k1))  # E[exp(-k t)]
    e2 = (np.exp(-2 * k1 * t) - np.exp(-2 * k2 * t)) / (2 * t * (k2 - k1))  # E[exp(-2 k t)]
    m_g, v_g = 1.0 - e1, e2 - e1**2
    m_a, v_a = 0.5 * (a1 + a2), (a2 - a1) ** 2 / 12.0
    V_k, V_a, V_ka = m_a**2 * v_g, m_g**2 * v_a, v_g * v_a
    V = V_k + V_a + V_ka
    return np.array([V_k, V_a]) / V, np.array([V_k + V_ka, V_a + V_ka]) / V


def test_reactor_decay_indices_match_closed_form(simple_model):
    """Through reactor.solve: first-order decay A -> B with the rate constant
    and the initial concentration uncertain. The estimate equals the same
    estimator applied to the closed-form solution on the same design (to the
    solve's tolerance), and both are within sampling error of the exact indices.
    """
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    p_def = simple_model.default_parameters()
    t_end = 10.0
    k_range, a_range = (0.05, 0.3), (0.5, 1.5)

    def fn(z):
        sol = reactor.solve(
            jnp.stack([z[1], 0.0]),
            params=p_def.at[0].set(z[0]),
            t_span=(0.0, t_end),
            t_eval=jnp.asarray([t_end]),
        )
        return sol.C_named("B")[-1]

    def exact_fn(z):
        return z[1] * (1.0 - jnp.exp(-z[0] * t_end))

    kw = dict(input_names=["k", "A0"], n_samples=256)
    res = aquakin.sobol_indices(fn, [k_range, a_range], **kw)
    ref = aquakin.sobol_indices(exact_fn, [k_range, a_range], **kw)
    # The solve's rtol is 1e-6; the indices are ratios of sample moments of it.
    np.testing.assert_allclose(res.first_order, ref.first_order, rtol=1e-4)
    np.testing.assert_allclose(res.total_order, ref.total_order, rtol=1e-4)
    S, ST = _decay_indices(t_end, k_range, a_range)
    # abs 0.02: the iid standard error at N = 256 is about 0.03 here (bootstrap
    # half-widths); scrambled Sobol does better (errors here < 6e-3).
    np.testing.assert_allclose(res.first_order, S, rtol=0, atol=0.02)
    np.testing.assert_allclose(res.total_order, ST, rtol=0, atol=0.02)


def test_vector_output_gives_one_result_per_output():
    """A vector fn gives one SobolResult per output, each with the indices of
    that output alone; per-point evaluation gives the same numbers."""
    fn = lambda z: jnp.stack([z[0], z[0] + 3.0 * z[1]])
    rng = [(0.0, 1.0), (0.0, 1.0)]
    res = aquakin.sobol_indices(fn, rng, output_names=["p", "q"], n_samples=256)
    assert [r.output_name for r in res] == ["p", "q"]
    np.testing.assert_allclose(res[0].first_order, [1.0, 0.0], rtol=0, atol=1e-3)
    np.testing.assert_allclose(res[1].first_order, [0.1, 0.9], rtol=0, atol=0.01)
    one_by_one = aquakin.sobol_indices(
        fn, rng, output_names=["p", "q"], n_samples=256, batched=False
    )
    for a, b in zip(res, one_by_one):
        np.testing.assert_allclose(a.first_order, b.first_order, rtol=1e-12)
        np.testing.assert_allclose(a.total_order_ci, b.total_order_ci, rtol=1e-12)


def test_non_finite_rows_are_dropped_per_output():
    """A row non-finite in one output is dropped from that output only."""
    fn = lambda z: jnp.stack([jnp.where(z[0] > 0.9, jnp.nan, z[0]), z[1]])
    res = aquakin.sobol_indices(fn, [(0.0, 1.0), (0.0, 1.0)], n_samples=256)
    assert res[0].n_valid < 256
    assert res[1].n_valid == 256
    np.testing.assert_allclose(res[1].first_order, [0.0, 1.0], rtol=0, atol=1e-3)


def test_indices_do_not_change_when_a_constant_is_added_to_the_output():
    """Sobol indices are invariant to f -> f + c, and so is the estimator: the
    first-order estimate centres the output, so an offset of 1e3 (as from a
    concentration reported in µmol/L on a large background) changes neither the
    estimate nor its interval beyond round-off."""
    rng = [(-PI, PI)] * 3
    base = aquakin.sobol_indices(_ishigami, rng, n_samples=256)
    shifted = aquakin.sobol_indices(lambda x: _ishigami(x) + 1e3, rng, n_samples=256)
    # Round-off of the 1e3 offset in ~1e1-sized differences of float64 values.
    np.testing.assert_allclose(shifted.first_order, base.first_order, rtol=0, atol=1e-9)
    np.testing.assert_allclose(shifted.first_order_ci, base.first_order_ci, rtol=0, atol=1e-9)
    np.testing.assert_allclose(shifted.total_order, base.total_order, rtol=0, atol=1e-9)


def test_seed_makes_the_result_reproducible():
    rng = [(0.0, 1.0), (0.0, 1.0)]
    fn = lambda z: jnp.sin(3.0 * z[0]) * z[1]
    a = aquakin.sobol_indices(fn, rng, n_samples=64, seed=3)
    b = aquakin.sobol_indices(fn, rng, n_samples=64, seed=3)
    c = aquakin.sobol_indices(fn, rng, n_samples=64, seed=4)
    assert np.array_equal(a.first_order, b.first_order)
    assert np.array_equal(a.first_order_ci, b.first_order_ci)
    assert not np.array_equal(a.first_order, c.first_order)


def test_zero_variance_output_warns_and_reports_zero():
    with pytest.warns(UserWarning, match="zero variance"):
        res = aquakin.sobol_indices(
            lambda z: jnp.asarray(2.0) + 0.0 * z[0], [(0.0, 1.0)], n_samples=16
        )
    assert res.output_variance == 0.0
    assert np.all(res.first_order == 0.0) and np.all(res.total_order_ci == 0.0)
    assert res.n_bootstrap == 0  # no intervals were formed


def test_bootstrap_resample_count_is_honoured_and_recorded():
    """The resample count changes the intervals (more resamples draw more rows
    from the same seeded stream) but not the point estimates, and each result
    records the count used; the default is 1000 at 95 %."""
    rng = [(-PI, PI)] * 3
    default = aquakin.sobol_indices(_ishigami, rng, n_samples=128)
    few = aquakin.sobol_indices(
        _ishigami, rng, n_samples=128, bootstrap=aquakin.BootstrapConfig(n_resamples=200)
    )
    assert (default.n_bootstrap, default.confidence) == (1000, 0.95)
    assert few.n_bootstrap == 200
    np.testing.assert_array_equal(few.first_order, default.first_order)
    np.testing.assert_array_equal(few.total_order, default.total_order)
    assert not np.array_equal(few.first_order_ci, default.first_order_ci)
    assert not np.array_equal(few.total_order_ci, default.total_order_ci)


def test_bootstrap_confidence_sets_the_interval_level():
    """With the same seed and resample count the bootstrap distribution is the
    same, so a 50 % interval lies inside the 95 % one, and the level is recorded."""
    rng = [(-PI, PI)] * 3
    wide = aquakin.sobol_indices(_ishigami, rng, n_samples=128)
    narrow = aquakin.sobol_indices(
        _ishigami, rng, n_samples=128, bootstrap=aquakin.BootstrapConfig(confidence=0.5)
    )
    assert narrow.confidence == 0.5
    for n_ci, w_ci in (
        (narrow.first_order_ci, wide.first_order_ci),
        (narrow.total_order_ci, wide.total_order_ci),
    ):
        assert np.all(w_ci[:, 0] < n_ci[:, 0]) and np.all(n_ci[:, 1] < w_ci[:, 1])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(n_resamples=99), "n_resamples must be an integer of at least 100; got 99"),
        (dict(n_resamples=500.0), "n_resamples must be an integer of at least 100; got 500.0"),
        (dict(n_resamples=True), "n_resamples must be an integer of at least 100; got True"),
        (dict(confidence=1.0), "confidence must lie strictly between 0 and 1; got 1.0"),
        (dict(confidence=0.0), "confidence must lie strictly between 0 and 1; got 0.0"),
    ],
)
def test_bootstrap_config_rejects_bad_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        aquakin.BootstrapConfig(**kwargs)


def test_sobol_indices_cost_rounds_to_a_power_of_two():
    """N rounds to the nearest power of two, exactly as the sampler draws."""
    assert aquakin.sobol_indices_cost(17, 1000) == 1024 * 19
    res = aquakin.sobol_indices(lambda z: z[0] + z[1], [(0.0, 1.0)] * 2, n_samples=30)
    assert res.n_samples == 32
    assert res.n_evaluations == aquakin.sobol_indices_cost(2, 30) == 128


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(distributions=[]), "at least one input distribution"),
        (dict(input_names=["only"]), "input_names has 1 entries but there are 2"),
        (dict(output_names=["a", "b"]), "output_names has 2 entries but fn returns m=1"),
        (dict(distributions=[(1.0, 0.0), (0.0, 1.0)]), "uniform needs high > low"),
    ],
)
def test_sobol_indices_rejects_bad_arguments(kwargs, match):
    args = dict(distributions=[(0.0, 1.0), (0.0, 1.0)], n_samples=8)
    args.update(kwargs)
    vector_fn = lambda z: jnp.stack([z[0] + z[-1]])
    with pytest.raises(ValueError, match=match):
        aquakin.sobol_indices(vector_fn, args.pop("distributions"), **args)


def test_an_all_non_finite_output_raises_and_narrowing_the_inputs_fixes_it():
    """fn is undefined (NaN) above z0 = 0.5: sampling only there raises and,
    following the message's advice, narrowing the input to where fn is defined
    gives the exact indices (z0 explains all the variance, z1 none)."""
    fn = lambda z: jnp.where(z[0] > 0.5, jnp.nan, z[0]) + 0.0 * z[1]
    with pytest.raises(RuntimeError, match="narrow them to where the model is defined"):
        aquakin.sobol_indices(fn, [(0.6, 1.0), (0.0, 1.0)], n_samples=8)
    res = aquakin.sobol_indices(fn, [(0.0, 0.5), (0.0, 1.0)], n_samples=64)
    # fn does not depend on z1, so fAB_1 == fA exactly and both z1 indices are
    # exactly 0. z0's are 1; abs 0.02 is well inside the bootstrap half-width at
    # N = 64 (about 0.27), which scrambled Sobol beats on a linear function.
    assert res.first_order[1] == 0.0 and res.total_order[1] == 0.0
    assert res.first_order[0] == pytest.approx(1.0, abs=0.02)
    assert res.total_order[0] == pytest.approx(1.0, abs=0.02)


def test_sobol_indices_cost_rejects_no_inputs():
    with pytest.raises(ValueError, match="n_inputs must be at least 1"):
        aquakin.sobol_indices_cost(0)
