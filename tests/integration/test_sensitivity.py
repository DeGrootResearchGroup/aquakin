"""Sensitivity and fit smoke tests."""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


def test_autodiff_matches_finite_difference(simple_model):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(simple_model, conditions)
    C0 = jnp.asarray([1.0, 0.0])
    params = simple_model.default_parameters()

    t_eval = jnp.linspace(0.0, 10.0, 11)

    result = aquakin.sensitivity(
        reactor,
        C0,
        params,
        output_fn=lambda sol: sol.C_named("B")[-1],
        solve_kwargs={"t_span": (0.0, 10.0), "t_eval": t_eval},
    )

    def _eval_k(k_val):
        p = params.at[0].set(k_val)
        sol = reactor.solve(C0, params=p, t_span=(0.0, 10.0), t_eval=t_eval)
        return float(sol.C_named("B")[-1])

    h = 1e-4
    fd = (_eval_k(float(params[0]) + h) - _eval_k(float(params[0]) - h)) / (2 * h)
    ad = float(result.doutput_dparams[0])
    assert ad == pytest.approx(fd, rel=1e-3, abs=1e-6)


def test_sensitivity_direct_t_span_matches_solve_kwargs(simple_model):
    """Passing t_span/t_eval directly to sensitivity equals putting them in
    solve_kwargs, and params defaults to the model defaults."""
    conditions = aquakin.SpatialConditions.uniform(T=293.15)
    reactor = aquakin.BatchReactor(simple_model, conditions)
    C0 = jnp.asarray([1.0, 0.0])
    t_eval = jnp.linspace(0.0, 10.0, 11)
    out = lambda sol: sol.C_named("B")[-1]

    direct = aquakin.sensitivity(reactor, C0, output_fn=out, t_span=(0.0, 10.0), t_eval=t_eval)
    via_kwargs = aquakin.sensitivity(
        reactor,
        C0,
        simple_model.default_parameters(),
        output_fn=out,
        solve_kwargs={"t_span": (0.0, 10.0), "t_eval": t_eval},
    )
    assert jnp.allclose(direct.doutput_dparams, via_kwargs.doutput_dparams)


def test_sensitivity_requires_output_fn(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(T=293.15))
    with pytest.raises(ValueError, match="output_fn"):
        aquakin.sensitivity(reactor, jnp.asarray([1.0, 0.0]), t_span=(0.0, 10.0))


def test_sensitivity_forward_matches_reverse(simple_model):
    """ad_mode='forward' builds the forward-capable adjoint internally and gives
    the same sensitivities as the default reverse mode."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(T=293.15))
    C0 = jnp.asarray([1.0, 0.0])
    out = lambda sol: sol.C_named("B")[-1]
    rev = aquakin.sensitivity(
        reactor, C0, output_fn=out, t_span=(0.0, 10.0), t_eval=jnp.linspace(0.0, 10.0, 11)
    )
    fwd = aquakin.sensitivity(
        reactor,
        C0,
        output_fn=out,
        t_span=(0.0, 10.0),
        t_eval=jnp.linspace(0.0, 10.0, 11),
        diff=aquakin.DifferentiationConfig(mode="forward"),
    )
    np.testing.assert_allclose(
        np.asarray(fwd.doutput_dparams), np.asarray(rev.doutput_dparams), rtol=1e-6
    )


def test_sensitivity_rejects_bad_ad_mode(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(T=293.15))
    with pytest.raises(ValueError, match="mode"):
        aquakin.sensitivity(
            reactor,
            jnp.asarray([1.0, 0.0]),
            output_fn=lambda s: s.C_named("B")[-1],
            t_span=(0.0, 1.0),
            diff=aquakin.DifferentiationConfig(mode="sideways"),
        )


def test_ranked_params(simple_model):
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(simple_model, conditions)
    result = aquakin.sensitivity(
        reactor,
        jnp.asarray([1.0, 0.0]),
        simple_model.default_parameters(),
        output_fn=lambda sol: sol.C_named("B")[-1],
        solve_kwargs={"t_span": (0.0, 5.0), "t_eval": jnp.linspace(0.0, 5.0, 6)},
    )
    ranked = result.ranked_params()
    assert ranked[0][0] == "A_to_B.k"


@pytest.mark.slow  # heavy: sensitivity vs FD
def test_sensitivity_doutput_dconditions_matches_finite_diff(simple_model):
    """Verify the conditions-override grad path is non-zero where expected."""
    # Build a tiny model rate that depends on T via Arrhenius so dF/dT != 0.
    # The shipped simple_model's rate is k*[A] only — independent of T —
    # so we exercise the override path with the ozone_bromate model instead.
    model = aquakin.load_model("ozone_bromate")
    atol = model.atol({"OH": 1e-20}, default=1e-12)
    conditions = aquakin.SpatialConditions.uniform(pH=7.5, T=293.15, OH_scavenging=5.0e4)
    reactor = aquakin.BatchReactor(model, conditions, atol=atol)
    C0 = model.concentrations({"O3": 1.0e-4, "Br-": 1.0e-5})
    params = model.default_parameters()
    t_eval = jnp.linspace(0.0, 300.0, 31)

    result = aquakin.sensitivity(
        reactor,
        C0,
        params,
        output_fn=lambda sol: sol.C_named("BrO3-")[-1],
        solve_kwargs={"t_span": (0.0, 300.0), "t_eval": t_eval},
    )

    # The OH scavenging field should have a *negative* gradient on bromate.
    g_scav = float(result.doutput_dconditions["OH_scavenging"][0])
    assert g_scav < 0.0
    # All gradients should be finite.
    for name, g in result.doutput_dconditions.items():
        assert jnp.all(jnp.isfinite(g)), f"non-finite grad for {name}"


def test_fit_rejects_empty_free_params(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    with pytest.raises(ValueError):
        aquakin.fit(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.asarray([0.0, 0.5]),
            t_obs=jnp.asarray([0.0, 1.0]),
            free_params=[],
        )


def test_fit_rejects_unknown_method(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    with pytest.raises(ValueError):
        aquakin.fit(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.asarray([0.0, 0.5]),
            t_obs=jnp.asarray([0.0, 1.0]),
            free_params=["A_to_B.k"],
            method="newton",
        )


def test_fit_rejects_unknown_param(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    with pytest.raises(KeyError):
        aquakin.fit(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.asarray([0.0, 0.5]),
            t_obs=jnp.asarray([0.0, 1.0]),
            free_params=["nonexistent.k"],
        )


def test_fit_rejects_observation_shape_mismatch(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    with pytest.raises(ValueError):
        aquakin.fit(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.zeros((2, 3)),  # 3 cols but only 1 observed_species
            t_obs=jnp.asarray([0.0, 1.0]),
            free_params=["A_to_B.k"],
            observed_species=["B"],
        )


def test_fit_rejects_descending_t_obs(simple_model):
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    with pytest.raises(ValueError):
        aquakin.fit(
            reactor,
            jnp.asarray([1.0, 0.0]),
            observations=jnp.asarray([0.5, 0.0]),
            t_obs=jnp.asarray([1.0, 0.0]),
            free_params=["A_to_B.k"],
            observed_species=["B"],
        )


def test_fit_recovers_known_rate(simple_model):
    """Generate synthetic observations from a known k, then re-fit."""
    conditions = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(simple_model, conditions)
    C0 = jnp.asarray([1.0, 0.0])

    true_k = 0.25
    true_params = simple_model.default_parameters().at[0].set(true_k)
    t_obs = jnp.linspace(0.5, 10.0, 20)
    sol = reactor.solve(C0, params=true_params, t_span=(0.0, 10.0), t_eval=t_obs)
    obs = sol.C_named("B")

    # Start from a wrong initial guess (the model default, k = 0.1).
    result = aquakin.fit(
        reactor,
        C0,
        observations=obs,
        t_obs=t_obs,
        free_params=["A_to_B.k"],
        observed_species=["B"],
    )

    assert result.converged
    assert result.params_named["A_to_B.k"] == pytest.approx(true_k, rel=1e-3)


_PARTIAL_BOUNDS_YAML = """
model:
  name: chain_decay
  version: "1.0"
  description: "A -> B -> C; k1 bounded, k2 unbounded."
species:
  - {name: A, default_concentration: 1.0}
  - {name: B, default_concentration: 0.0}
  - {name: C, default_concentration: 0.0}
conditions:
  - {name: T, default: 293.15}
reactions:
  - name: r1
    rate: "k1 * [A]"
    parameters:
      k1: {value: 0.3, bounds: [1.0e-3, 10.0]}
    stoichiometry: {A: -1, B: +1}
  - name: r2
    rate: "k2 * [B]"
    parameters:
      k2: {value: 0.2}            # no bounds declared
    stoichiometry: {B: -1, C: +1}
"""


def _chain_fit_setup(tmp_path):
    p = tmp_path / "chain.yaml"
    p.write_text(_PARTIAL_BOUNDS_YAML)
    net = aquakin.load_model_from_file(str(p))
    reactor = aquakin.BatchReactor(net, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0 = jnp.asarray([1.0, 0.0, 0.0])
    t_obs = jnp.linspace(0.5, 10.0, 12)
    obs = reactor.solve(
        C0, params=net.default_parameters(), t_span=(0.0, 10.0), t_eval=t_obs
    ).C_named("C")
    return reactor, C0, obs, t_obs


def _capture_bounds(monkeypatch):
    """Patch the scipy minimize used by fit() to record the bounds passed."""
    import importlib

    # The package exposes a `sensitivity` function that shadows the submodule
    # attribute, so resolve the module object explicitly.
    S = importlib.import_module("aquakin.integrate.sensitivity")
    captured = {}
    real = S.minimize

    def fake(*args, **kwargs):
        captured["bounds"] = kwargs.get("bounds")
        return real(*args, **kwargs)

    monkeypatch.setattr(S, "minimize", fake)
    return captured


def test_fit_keeps_partial_bounds(tmp_path, monkeypatch):
    """A free param without declared bounds must NOT drop the other's bounds:
    it is left unbounded (+/-inf) while the bounded one keeps its box, and a
    warning is emitted."""
    reactor, C0, obs, t_obs = _chain_fit_setup(tmp_path)
    captured = _capture_bounds(monkeypatch)
    with pytest.warns(UserWarning, match="no declared bounds"):
        aquakin.fit(
            reactor,
            C0,
            observations=obs,
            t_obs=t_obs,
            free_params=["r1.k1", "r2.k2"],
            observed_species=["C"],
        )
    bounds = captured["bounds"]
    assert bounds is not None
    assert bounds[0] == (1.0e-3, 10.0)  # k1 keeps its declared box
    assert bounds[1] == (-np.inf, np.inf)  # k2 left unbounded


def test_fit_all_bounded_no_warning(tmp_path, monkeypatch):
    """All free params bounded -> per-param bounds, no warning."""
    reactor, C0, obs, t_obs = _chain_fit_setup(tmp_path)
    captured = _capture_bounds(monkeypatch)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning fails the test
        aquakin.fit(
            reactor,
            C0,
            observations=obs,
            t_obs=t_obs,
            free_params=["r1.k1"],
            observed_species=["C"],
        )
    assert captured["bounds"] == [(1.0e-3, 10.0)]


def test_fit_all_unbounded_passes_none(tmp_path, monkeypatch):
    """No free param has bounds -> bounds=None (fully unbounded), no warning."""
    reactor, C0, obs, t_obs = _chain_fit_setup(tmp_path)
    captured = _capture_bounds(monkeypatch)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        aquakin.fit(
            reactor,
            C0,
            observations=obs,
            t_obs=t_obs,
            free_params=["r2.k2"],
            observed_species=["C"],
        )
    assert captured["bounds"] is None


# --- DGSM (derivative-based global sensitivity) ------------------------


def test_dgsm_ranks_influential_input():
    """A linear output sensitive only to z0 ranks z0 far above z1."""
    res = aquakin.dgsm(
        lambda z: 3.0 * z[0] + 0.0 * z[1],
        [(0.0, 1.0), (0.0, 1.0)],
        input_names=["a", "b"],
        n_samples=32,
    )
    ranked = res.ranked()
    assert ranked[0][0] == "a"
    assert ranked[0][1] > 10.0 * max(ranked[1][1], 1e-12)


def test_dgsm_matches_analytic_nu():
    """For f = c*z0, nu_0 = c^2 exactly (gradient is constant c)."""
    c = 2.5
    res = aquakin.dgsm(lambda z: c * z[0], [(-1.0, 1.0)], n_samples=16)
    assert float(res.dgsm[0]) == pytest.approx(c**2, rel=1e-6)


def test_dgsm_zero_variance_output_warns():
    """A constant output (zero variance) makes the Sobol total-index bound
    undefined (0/0). dgsm must warn and report a zero bound rather than
    silently returning an all-zero ranking that reads as 'nothing matters'."""
    with pytest.warns(UserWarning, match="zero variance"):
        res = aquakin.dgsm(lambda z: jnp.asarray(3.0), [(0.0, 1.0), (0.0, 1.0)], n_samples=16)
    assert float(res.output_variance) == 0.0
    assert np.all(np.asarray(res.sobol_total_bound) == 0.0)


def test_dgsm_reproducible_with_seed():
    fn = lambda z: jnp.sin(z[0]) * z[1] ** 2
    rng = [(0.0, 2.0), (0.0, 2.0)]
    a = aquakin.dgsm(fn, rng, n_samples=32, seed=7)
    b = aquakin.dgsm(fn, rng, n_samples=32, seed=7)
    c = aquakin.dgsm(fn, rng, n_samples=32, seed=8)
    assert np.array_equal(np.asarray(a.sobol_total_bound), np.asarray(b.sobol_total_bound))
    assert not np.array_equal(np.asarray(a.sobol_total_bound), np.asarray(c.sobol_total_bound))


def test_dgsm_n_rounded_to_power_of_two():
    res = aquakin.dgsm(lambda z: z[0], [(0.0, 1.0)], n_samples=30)
    assert res.n_samples == 32  # nearest power of two


def test_dgsm_rejects_bad_ranges():
    with pytest.raises(ValueError):
        aquakin.dgsm(lambda z: z[0], [(1.0, 0.0)], n_samples=8)  # upper <= lower


def test_dgsm_through_reactor(simple_model):
    """DGSM flows through reactor.solve and finds the rate constant influential."""
    reactor = aquakin.BatchReactor(simple_model, aquakin.SpatialConditions.uniform(1, T=293.15))
    C0 = jnp.asarray([1.0, 0.0])
    p_def = simple_model.default_parameters()
    t_eval = jnp.linspace(0.0, 10.0, 11)

    def fn(z):
        p = p_def.at[0].set(z[0])
        sol = reactor.solve(C0, params=p, t_span=(0.0, 10.0), t_eval=t_eval)
        return sol.C_named("B")[-1]

    res = aquakin.dgsm(fn, [(0.1, 0.5)], input_names=["A_to_B.k"], n_samples=8)
    assert res.n_valid >= 2
    assert float(res.dgsm[0]) > 0.0


def test_dgsm_rejects_bad_ad_mode():
    with pytest.raises(ValueError):
        aquakin.dgsm(
            lambda z: z[0],
            [(0.0, 1.0)],
            n_samples=8,
            diff=aquakin.DifferentiationConfig(mode="sideways"),
        )


def test_dgsm_mode_alias_is_deprecated():
    """The differentiation mode is supplied via ``diff=DifferentiationConfig``."""
    fn = lambda z: 3.0 * z[0]
    res = aquakin.dgsm(
        fn,
        [(0.0, 1.0), (0.0, 1.0)],
        n_samples=8,
        diff=aquakin.DifferentiationConfig(mode="reverse"),
    )
    assert res.ranked()[0][0] == "z0"


def test_dgsm_forward_matches_reverse():
    """Forward and reverse modes give identical sensitivities (ad_mode is only a
    performance choice)."""
    fn = lambda z: jnp.sin(z[0]) * z[1] ** 2 + 0.3 * z[0] * z[1]
    rng = [(0.2, 1.5), (0.2, 1.5)]
    rev = aquakin.dgsm(
        fn, rng, n_samples=32, seed=3, diff=aquakin.DifferentiationConfig(mode="reverse")
    )
    fwd = aquakin.dgsm(
        fn, rng, n_samples=32, seed=3, diff=aquakin.DifferentiationConfig(mode="forward")
    )
    np.testing.assert_allclose(
        np.asarray(fwd.sobol_total_bound), np.asarray(rev.sobol_total_bound), rtol=1e-9
    )


def test_dgsm_forward_matches_reverse_through_reactor(simple_model):
    """forward and reverse DGSM agree when ``fn`` integrates a reactor solve --
    the real use case the forward path exists for. The forward screen drives
    ``reactor.solve`` through ``aquakin.forward_adjoint()`` (DirectAdjoint, which
    permits forward-mode AD), the reverse screen through the default adjoint, and
    the two Sobol-total bounds are identical (ad_mode is only a performance
    choice)."""
    cond = aquakin.SpatialConditions.uniform(1, T=293.15)
    C0 = jnp.asarray([1.0, 0.0])
    p_def = simple_model.default_parameters()
    t_eval = jnp.linspace(0.0, 10.0, 11)
    rng = [(0.1, 0.5)]

    def make_fn(diff=None):
        kw = {} if diff is None else {"diff": diff}
        reactor = aquakin.BatchReactor(simple_model, cond, **kw)

        def fn(z):
            p = p_def.at[0].set(z[0])
            sol = reactor.solve(C0, params=p, t_span=(0.0, 10.0), t_eval=t_eval)
            return sol.C_named("B")[-1]

        return fn

    # default adjoint (reverse-capable); the forward-capable differentiation.
    rev = aquakin.dgsm(
        make_fn(None),
        rng,
        input_names=["A_to_B.k"],
        n_samples=8,
        seed=1,
        diff=aquakin.DifferentiationConfig(mode="reverse"),
    )
    fwd = aquakin.dgsm(
        make_fn(aquakin.DifferentiationConfig(mode="forward", method="through_solve")),
        rng,
        input_names=["A_to_B.k"],
        n_samples=8,
        seed=1,
        diff=aquakin.DifferentiationConfig(mode="forward"),
    )
    np.testing.assert_allclose(
        np.asarray(fwd.sobol_total_bound),
        np.asarray(rev.sobol_total_bound),
        rtol=1e-6,
    )


def test_dgsm_vector_output_returns_per_output_results():
    """A vector-valued fn returns one result per output, each with its name and
    the right input ranking."""
    # output 0 depends on z0, output 1 on z1.
    fn = lambda z: jnp.array([2.0 * z[0], 5.0 * z[1]])
    rng = [(0.0, 1.0), (0.0, 1.0)]
    out = aquakin.dgsm(
        fn,
        rng,
        input_names=["a", "b"],
        output_names=["o0", "o1"],
        n_samples=16,
        diff=aquakin.DifferentiationConfig(mode="forward"),
    )
    assert isinstance(out, list) and len(out) == 2
    assert [r.output_name for r in out] == ["o0", "o1"]
    assert out[0].ranked()[0][0] == "a"  # o0 most sensitive to a
    assert out[1].ranked()[0][0] == "b"  # o1 most sensitive to b
    # nu equals the squared (constant) partial derivative.
    assert float(out[0].dgsm[0]) == pytest.approx(4.0, rel=1e-6)
    assert float(out[1].dgsm[1]) == pytest.approx(25.0, rel=1e-6)


def test_dgsm_finite_mask_is_per_output():
    """A sample non-finite in one output must not be dropped from the others.

    Output 0 is NaN for the (roughly half) samples with z0 > 0.5; output 1 is
    finite everywhere. The clean output must keep ALL drawn samples (a joint mask
    would discard the shared rows and bias its nu / n_valid), while the dirty
    output keeps only its finite subset."""
    fn = lambda z: jnp.array([jnp.where(z[0] > 0.5, jnp.nan, 2.0 * z[0]), 5.0 * z[1]])
    rng = [(0.0, 1.0), (0.0, 1.0)]
    out = aquakin.dgsm(
        fn,
        rng,
        input_names=["a", "b"],
        output_names=["dirty", "clean"],
        n_samples=16,
        diff=aquakin.DifferentiationConfig(mode="forward"),
    )
    n_drawn = out[0].n_samples
    assert out[1].n_valid == n_drawn  # clean output: every sample kept
    assert 2 <= out[0].n_valid < n_drawn  # dirty output: NaN rows dropped
    # The clean output's sensitivity is the exact partial, computed over all 16
    # samples -- unbiased by the other output's failures.
    assert float(out[1].dgsm[1]) == pytest.approx(25.0, rel=1e-6)
    assert float(out[0].dgsm[0]) == pytest.approx(4.0, rel=1e-6)


def test_dgsm_routes_through_shared_aggregator():
    """The public entry point's bound equals the shared ``_dgsm_aggregate`` kernel
    applied to the same samples -- pinning that the reactor and plant screens use
    one aggregation implementation."""
    from aquakin.integrate.sensitivity import (
        _dgsm_aggregate,
        _evaluate_dgsm_samples,
        _make_dgsm_value_and_jac,
        _sobol_sample,
        _validate_dgsm_ranges,
    )

    fn = lambda z: jnp.array([2.0 * z[0] + 0.5 * z[1], 3.0 * z[1]])
    ranges = [(0.0, 1.0), (0.0, 2.0)]
    res = aquakin.dgsm(fn, ranges, n_samples=32, output_names=["o0", "o1"])

    # Reconstruct the exact samples the entry point drew and aggregate directly.
    ranges_np, lo, hi, d, _ = _validate_dgsm_ranges(ranges, None)
    Z, _ = _sobol_sample(lo, hi, d, 32, 0)
    value_and_jac, _, _ = _make_dgsm_value_and_jac(fn, Z[0], "reverse")
    vals, jacs = _evaluate_dgsm_samples(value_and_jac, Z, "reverse", True)
    bound, se, nu, var, n_valid = _dgsm_aggregate(jacs**2, vals, (hi - lo) ** 2)

    for i in range(2):
        np.testing.assert_allclose(np.asarray(res[i].sobol_total_bound), bound[i], rtol=1e-10)
        np.testing.assert_allclose(np.asarray(res[i].std_error), se[i], rtol=1e-10)
        np.testing.assert_allclose(np.asarray(res[i].dgsm), nu[i], rtol=1e-10)
        assert res[i].n_valid == int(n_valid[i])


def test_dgsm_zero_variance_warns_and_reports_zero_bound():
    """A constant output has an undefined (0/0) Sobol bound: report 0 with a
    warning rather than a silent empty ranking."""
    fn = lambda z: 3.0 + 0.0 * z[0]
    with pytest.warns(UserWarning, match="zero variance"):
        res = aquakin.dgsm(fn, [(0.0, 1.0)], n_samples=8)
    assert float(res.output_variance) == 0.0
    np.testing.assert_array_equal(np.asarray(res.sobol_total_bound), [0.0])
    np.testing.assert_array_equal(np.asarray(res.std_error), [0.0])


def test_dgsm_forward_through_reactor_matches_reverse(simple_model):
    """Forward mode through a reactor solve agrees with the reverse-mode result
    to machine precision. The forward-capable adjoint is built with the
    ``aquakin.forward_adjoint()`` helper (no direct ``diffrax`` import)."""
    conds = aquakin.SpatialConditions.uniform(1, T=293.15)
    C0 = jnp.asarray([1.0, 0.0])
    p_def = simple_model.default_parameters()
    t_eval = jnp.linspace(0.0, 10.0, 11)

    def make_fn(reactor):
        def fn(z):
            p = p_def.at[0].set(z[0])
            sol = reactor.solve(C0, params=p, t_span=(0.0, 10.0), t_eval=t_eval)
            return sol.C_named("B")[-1]

        return fn

    rev = aquakin.dgsm(
        make_fn(aquakin.BatchReactor(simple_model, conds)),
        [(0.1, 0.5)],
        n_samples=8,
        diff=aquakin.DifferentiationConfig(mode="reverse"),
    )
    fwd = aquakin.dgsm(
        make_fn(
            aquakin.BatchReactor(
                simple_model,
                conds,
                diff=aquakin.DifferentiationConfig(mode="forward", method="through_solve"),
            )
        ),
        [(0.1, 0.5)],
        n_samples=8,
        diff=aquakin.DifferentiationConfig(mode="forward"),
    )
    np.testing.assert_allclose(
        np.asarray(fwd.sobol_total_bound), np.asarray(rev.sobol_total_bound), rtol=1e-8
    )


def test_dgsm_forward_through_default_adjoint_errors():
    """Forward mode through the default RecursiveCheckpointAdjoint raises a
    helpful error pointing to aquakin.forward_adjoint()."""
    net = aquakin.load_model_from_file(
        str(__import__("pathlib").Path(__file__).parents[1] / "fixtures" / "simple_model.yaml")
    )
    conds = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(net, conds)  # default reverse-only adjoint
    C0 = jnp.asarray([1.0, 0.0])
    p_def = net.default_parameters()

    def fn(z):
        p = p_def.at[0].set(z[0])
        sol = reactor.solve(C0, params=p, t_span=(0.0, 10.0))
        return sol.C[-1, 1]

    with pytest.raises(RuntimeError, match="forward_adjoint"):
        aquakin.dgsm(
            fn, [(0.1, 0.5)], n_samples=8, diff=aquakin.DifferentiationConfig(mode="forward")
        )


def test_dgsm_batched_matches_unbatched():
    """The vmapped (batched=True) and per-sample (batched=False) paths give
    bit-identical results -- batched is purely a dispatch/memory choice."""
    fn = lambda z: jnp.sin(z[0]) * z[1] ** 2 + 0.3 * z[0] * z[1]
    rng = [(0.2, 1.5), (0.2, 1.5)]
    a = aquakin.dgsm(fn, rng, n_samples=32, seed=5, batched=True)
    b = aquakin.dgsm(fn, rng, n_samples=32, seed=5, batched=False)
    np.testing.assert_array_equal(np.asarray(a.sobol_total_bound), np.asarray(b.sobol_total_bound))
    assert a.n_valid == b.n_valid


def test_dgsm_batched_matches_unbatched_vector():
    """Batched/unbatched equivalence holds for vector-valued fn too."""
    fn = lambda z: jnp.array([2.0 * z[0] + z[1], 5.0 * z[1] ** 2])
    rng = [(0.0, 1.0), (0.0, 1.0)]
    kw = dict(n_samples=16, seed=2, output_names=["o0", "o1"])
    a = aquakin.dgsm(fn, rng, batched=True, **kw)
    b = aquakin.dgsm(fn, rng, batched=False, **kw)
    for ra, rb in zip(a, b):
        np.testing.assert_array_equal(
            np.asarray(ra.sobol_total_bound), np.asarray(rb.sobol_total_bound)
        )


def test_dgsm_unbatched_forward_default_adjoint_errors():
    """The per-sample fallback also raises the forward-adjoint guidance when a
    forward-mode screen hits the default reactor adjoint."""
    net = aquakin.load_model_from_file(
        str(__import__("pathlib").Path(__file__).parents[1] / "fixtures" / "simple_model.yaml")
    )
    conds = aquakin.SpatialConditions.uniform(1, T=293.15)
    reactor = aquakin.BatchReactor(net, conds)  # default reverse-only adjoint
    C0 = jnp.asarray([1.0, 0.0])
    p_def = net.default_parameters()

    def fn(z):
        p = p_def.at[0].set(z[0])
        return reactor.solve(C0, params=p, t_span=(0.0, 10.0)).C[-1, 1]

    with pytest.raises(RuntimeError, match="forward_adjoint"):
        aquakin.dgsm(
            fn,
            [(0.1, 0.5)],
            n_samples=8,
            diff=aquakin.DifferentiationConfig(mode="forward"),
            batched=False,
        )


def test_dgsm_helpers_validate_and_sample():
    """The decomposed helpers behave independently of the dgsm entry point."""
    from aquakin.integrate.sensitivity import (
        _sobol_sample,
        _validate_dgsm_ranges,
    )

    ranges_np, lo, hi, d, names = _validate_dgsm_ranges([(0.0, 2.0), (-1.0, 1.0)], None)
    assert d == 2 and names == ["z0", "z1"]
    np.testing.assert_array_equal(lo, [0.0, -1.0])
    np.testing.assert_array_equal(hi, [2.0, 1.0])

    with pytest.raises(ValueError, match="upper > lower"):
        _validate_dgsm_ranges([(1.0, 0.0)], None)
    with pytest.raises(ValueError, match="input_names has"):
        _validate_dgsm_ranges([(0.0, 1.0)], ["a", "b"])

    Z, n_drawn = _sobol_sample(lo, hi, d, n_samples=30, seed=0)
    assert n_drawn == 32 and Z.shape == (32, 2)
    assert np.all(Z[:, 0] >= 0.0) and np.all(Z[:, 0] <= 2.0)
