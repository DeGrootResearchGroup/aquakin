"""Scenario comparison and Monte-Carlo uncertainty propagation
(aquakin.monte_carlo / aquakin.compare_scenarios).

Most cases use a closed-form ``fn`` (no ODE solve) to test the sampling,
distribution and aggregation logic quickly; one case runs a real reactor solve
to confirm the end-to-end path.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin


# A cheap differentiable model: two inputs -> two outputs, no solve.
def _fn(x):
    return jnp.array([x[0] + x[1], x[0] * x[1]])


def _fn_scalar(x):
    return x[0] ** 2 + x[1]


# --- monte_carlo -------------------------------------------------------------


def test_monte_carlo_basic_shapes_and_names():
    mc = aquakin.monte_carlo(
        _fn, {"a": (0.0, 1.0), "b": (2.0, 4.0)}, output_names=["sum", "prod"], n_samples=64, seed=0
    )
    assert mc.input_names == ["a", "b"]
    assert mc.output_names == ["sum", "prod"]
    assert mc.samples.shape == (64, 2)
    assert mc.outputs.shape == (64, 2)
    assert mc.n_valid == 64
    # sum = a + b is in [2, 5]; prod = a*b in [0, 4].
    assert np.all((mc.output_named("sum") >= 2.0) & (mc.output_named("sum") <= 5.0))
    assert mc.percentiles((50.0,)).shape == (1, 2)


def test_monte_carlo_is_reproducible_by_seed():
    kw = dict(distributions=[(0.0, 1.0), (0.0, 1.0)], n_samples=32)
    a = aquakin.monte_carlo(_fn, **kw, seed=7)
    b = aquakin.monte_carlo(_fn, **kw, seed=7)
    c = aquakin.monte_carlo(_fn, **kw, seed=8)
    assert np.array_equal(a.outputs, b.outputs)
    assert not np.array_equal(a.outputs, c.outputs)


@pytest.mark.parametrize("sampler", ["sobol", "lhs", "random"])
def test_monte_carlo_samplers_run(sampler):
    mc = aquakin.monte_carlo(_fn, [(0.0, 1.0), (0.0, 1.0)], n_samples=64, sampler=sampler, seed=1)
    assert mc.sampler == sampler
    assert mc.n_valid > 0
    assert np.all(np.isfinite(mc.outputs))


def test_monte_carlo_recovers_input_distribution_moments():
    # fn returns the inputs unchanged, so the sampled output marginals must match
    # the requested distributions' moments (a check that inverse-transform
    # sampling of normal / lognormal is wired correctly).
    mc = aquakin.monte_carlo(
        lambda x: x,
        {
            "normal": {"dist": "normal", "mean": 5.0, "std": 1.0},
            "lognorm": {"dist": "lognormal", "mean": 2.0, "std": 0.5},
        },
        output_names=["normal", "lognorm"],
        n_samples=4096,
        seed=0,
    )
    norm_col = mc.output_named("normal")
    ln_col = mc.output_named("lognorm")
    assert norm_col.mean() == pytest.approx(5.0, abs=0.1)
    assert norm_col.std() == pytest.approx(1.0, abs=0.1)
    assert ln_col.mean() == pytest.approx(2.0, rel=0.1)
    assert ln_col.std() == pytest.approx(0.5, rel=0.2)
    assert np.all(ln_col > 0.0)  # lognormal is positive


def test_monte_carlo_scalar_output_and_dropping_nonfinite():
    # fn returns inf when a > 0.9, so those rows are dropped.
    def fn(x):
        return jnp.where(x[0] > 0.9, jnp.inf, x[0] + x[1])

    mc = aquakin.monte_carlo(fn, [(0.0, 1.0), (0.0, 1.0)], n_samples=128, seed=0)
    assert mc.output_names == ["output"]
    assert mc.n_valid < mc.n_drawn  # some rows dropped
    assert np.all(np.isfinite(mc.outputs))


def test_monte_carlo_summary_is_a_table():
    mc = aquakin.monte_carlo(
        _fn, [(0.0, 1.0), (0.0, 1.0)], output_names=["sum", "prod"], n_samples=16
    )
    s = mc.summary()
    assert "Monte-Carlo" in s and "sum" in s and "prod" in s


def test_monte_carlo_bad_distribution_raises():
    with pytest.raises(ValueError, match="unknown distribution"):
        aquakin.monte_carlo(_fn, [{"dist": "weird", "x": 1}], n_samples=8)


# --- compare_scenarios -------------------------------------------------------


def test_compare_scenarios_overrides_on_baseline():
    sc = aquakin.compare_scenarios(
        _fn,
        {"base": {}, "hi_a": {"a": 1.0}, "hi_b": {"b": 10.0}},
        input_names=["a", "b"],
        baseline=[0.5, 2.0],
        output_names=["sum", "prod"],
    )
    assert sc.scenario_names == ["base", "hi_a", "hi_b"]
    # base: 0.5+2=2.5 ; hi_a: 1+2=3 ; hi_b: 0.5+10=10.5
    assert np.allclose(sc.output_named("sum"), [2.5, 3.0, 10.5])
    assert sc.best("sum", minimize=True) == "base"
    assert sc.best("sum", minimize=False) == "hi_b"


def test_compare_scenarios_full_vectors_and_table():
    sc = aquakin.compare_scenarios(
        _fn,
        {"x": [1.0, 1.0], "y": [2.0, 3.0]},
        input_names=["a", "b"],
        output_names=["sum", "prod"],
    )
    assert np.allclose(sc.output_named("prod"), [1.0, 6.0])
    assert "scenario" in sc.table() and "x" in sc.table()


def test_compare_scenarios_unknown_override_raises():
    with pytest.raises(KeyError, match="unknown input"):
        aquakin.compare_scenarios(
            _fn, {"s": {"ghost": 1.0}}, input_names=["a", "b"], baseline=[0.0, 0.0]
        )


def test_compare_scenarios_nonfinite_raises():
    def fn(x):
        return jnp.where(x[0] > 5.0, jnp.nan, x[0])

    with pytest.raises(ValueError, match="non-finite"):
        aquakin.compare_scenarios(
            fn, {"ok": {"a": 1.0}, "bad": {"a": 9.0}}, input_names=["a", "b"], baseline=[0.0, 0.0]
        )


# --- end-to-end with a real solve --------------------------------------------


def test_monte_carlo_through_a_reactor_solve():
    # The full path: fn builds params and runs a stiff ASM solve. Uncertain AOB
    # max growth rate -> effluent ammonia distribution.
    net = aquakin.load_model("asm3_2step")
    r = aquakin.BatchReactor(net, aquakin.SpatialConditions.uniform(T=293.15))
    C0 = net.concentrations({"SO2": 300.0, "SNH4": 30.0, "XAOB": 80.0, "XNOB": 80.0, "SALK": 0.05})
    i_mu = net.param_index["muAOB"]

    def fn(x):
        p = net.default_parameters().at[i_mu].set(x[0])
        sol = r.solve(C0, params=p, t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 6))
        return sol.C_named("SNH4")[-1]

    mc = aquakin.monte_carlo(fn, {"muAOB": (0.6, 1.2)}, n_samples=16, seed=0)
    assert mc.n_valid == 16
    assert np.all(np.isfinite(mc.outputs))
    # Faster AOB -> lower effluent ammonia, so the output spans a real range.
    assert mc.std()[0] > 0.0


# --- optimize_design ---------------------------------------------------------


def test_optimize_design_analytical_with_constraint():
    # min x0^2 + x1^2 s.t. x0 + x1 >= 1  ->  (0.5, 0.5), objective 0.5.
    res = aquakin.optimize_design(
        lambda x: x[0] ** 2 + x[1] ** 2,
        bounds=[(0.0, 2.0), (0.0, 2.0)],
        input_names=["x0", "x1"],
        constraints=[aquakin.Constraint(fn=lambda x: x[0] + x[1], lower=1.0, name="sum")],
    )
    assert res.success and res.feasible
    assert np.allclose(res.x, [0.5, 0.5], atol=1e-3)
    assert res.objective == pytest.approx(0.5, abs=1e-3)
    assert res.x_named["x0"] == pytest.approx(0.5, abs=1e-3)
    assert res.constraint_values["sum"] == pytest.approx(1.0, abs=1e-3)


def test_optimize_design_maximize():
    # max x0 s.t. x0 <= 3 (upper constraint) -> x0 = 3.
    res = aquakin.optimize_design(
        lambda x: x[0],
        bounds=[(0.0, 10.0)],
        maximize=True,
        constraints=[aquakin.Constraint(fn=lambda x: x[0], upper=3.0)],
    )
    assert res.feasible
    assert res.x[0] == pytest.approx(3.0, abs=1e-3)
    assert res.objective == pytest.approx(3.0, abs=1e-3)


def test_optimize_design_box_bounds_active():
    # Unconstrained min of (x - 5)^2 with the box capping x at 2 -> x = 2.
    res = aquakin.optimize_design(lambda x: (x[0] - 5.0) ** 2, bounds=[(0.0, 2.0)])
    assert res.x[0] == pytest.approx(2.0, abs=1e-3)


def test_optimize_design_reports_infeasible():
    # No x in [0, 1] satisfies x >= 5; the optimizer must report infeasible.
    res = aquakin.optimize_design(
        lambda x: x[0],
        bounds=[(0.0, 1.0)],
        constraints=[aquakin.Constraint(fn=lambda x: x[0], lower=5.0)],
    )
    assert not res.feasible


def test_optimize_design_multistart_runs():
    res = aquakin.optimize_design(
        lambda x: (x[0] - 1.0) ** 2 + (x[1] + 1.0) ** 2,
        bounds=[(-3.0, 3.0), (-3.0, 3.0)],
        n_starts=4,
        seed=0,
    )
    assert res.n_starts == 4
    assert np.allclose(res.x, [1.0, -1.0], atol=1e-2)


def test_constraint_needs_a_bound():
    with pytest.raises(ValueError, match="needs an 'upper'"):
        aquakin.Constraint(fn=lambda x: x[0])


def test_optimize_design_through_a_reactor_solve():
    # Size the AOB growth rate to a (feasible) effluent-ammonia permit at minimum
    # rate: minimise muAOB s.t. effluent NH4 <= 6.5 gN/m3. The constraint is
    # active at the optimum (the smallest muAOB that still meets the permit).
    net = aquakin.load_model("asm3_2step")
    r = aquakin.BatchReactor(net, aquakin.SpatialConditions.uniform(T=293.15))
    C0 = net.concentrations({"SO2": 300.0, "SNH4": 30.0, "XAOB": 80.0, "XNOB": 80.0, "SALK": 0.05})
    i_mu = net.param_index["muAOB"]

    def eff_nh4(x):
        p = net.default_parameters().at[i_mu].set(x[0])
        return r.solve(C0, params=p, t_span=(0.0, 1.0), t_eval=jnp.linspace(0.0, 1.0, 6)).C_named(
            "SNH4"
        )[-1]

    res = aquakin.optimize_design(
        objective=lambda x: x[0],
        bounds=[(0.5, 2.0)],
        input_names=["muAOB"],
        constraints=[aquakin.Constraint(fn=eff_nh4, upper=6.5, name="eff_NH4")],
        x0=[1.5],
    )
    assert res.feasible
    assert res.constraint_values["eff_NH4"] <= 6.5 + 1e-3  # permit met
    assert 0.5 <= res.x[0] <= 2.0


# --- argument-validation raises ----------------------------------------------
# These cover the input-validation guards in the monte_carlo / scenarios /
# design modules (the experiments public API). They are pure validation, so no
# solve is needed.


# monte_carlo distribution specs ------------------------------------------------


def test_monte_carlo_malformed_distribution_spec_raises():
    # A spec that is neither a (low, high) pair nor a {"dist": ...} mapping.
    with pytest.raises(ValueError, match=r"must be a \(low, high\) tuple or a mapping"):
        aquakin.monte_carlo(_fn, [{"low": 0.0, "high": 1.0}], n_samples=8)


def test_monte_carlo_uniform_requires_high_gt_low():
    with pytest.raises(ValueError, match="uniform needs high > low"):
        aquakin.monte_carlo(_fn, [(1.0, 1.0)], n_samples=8)


def test_monte_carlo_normal_requires_positive_std():
    with pytest.raises(ValueError, match="normal needs std > 0"):
        aquakin.monte_carlo(_fn, [{"dist": "normal", "mean": 0.0, "std": 0.0}], n_samples=8)


def test_monte_carlo_lognormal_requires_positive_mean_and_std():
    with pytest.raises(ValueError, match="lognormal needs mean > 0 and std > 0"):
        aquakin.monte_carlo(_fn, [{"dist": "lognormal", "mean": -1.0, "std": 0.5}], n_samples=8)


def test_monte_carlo_input_names_length_mismatch_raises():
    with pytest.raises(ValueError, match="input_names has 1 entries but there are 2"):
        aquakin.monte_carlo(_fn, [(0.0, 1.0), (0.0, 1.0)], input_names=["only_one"], n_samples=8)


def test_monte_carlo_output_named_unknown_raises():
    mc = aquakin.monte_carlo(
        _fn, [(0.0, 1.0), (0.0, 1.0)], output_names=["sum", "prod"], n_samples=8
    )
    with pytest.raises(KeyError, match="unknown output 'ghost'"):
        mc.output_named("ghost")


# scenarios --------------------------------------------------------------------


def test_compare_scenarios_baseline_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"baseline must have shape \(2,\)"):
        aquakin.compare_scenarios(
            _fn, {"s": {"a": 1.0}}, input_names=["a", "b"], baseline=[0.0, 0.0, 0.0]
        )


def test_compare_scenarios_full_vector_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"vector must have shape \(2,\)"):
        aquakin.compare_scenarios(_fn, {"s": [1.0]}, input_names=["a", "b"])


def test_scenario_comparison_output_named_unknown_raises():
    sc = aquakin.compare_scenarios(
        _fn, {"x": [1.0, 1.0]}, input_names=["a", "b"], output_names=["sum", "prod"]
    )
    with pytest.raises(KeyError, match="unknown output 'ghost'"):
        sc.output_named("ghost")


def test_kpi_comparison_column_unknown_raises():
    kc = aquakin.kpi_comparison({"a": {"EQI": 1.0}, "b": {"EQI": 2.0}})
    with pytest.raises(KeyError, match="unknown KPI 'ghost'"):
        kc.column("ghost")


def test_kpi_comparison_best_no_finite_value_raises():
    kc = aquakin.kpi_comparison({"a": {"EQI": float("nan")}})
    with pytest.raises(ValueError, match="no finite value"):
        kc.best("EQI")


def test_kpi_comparison_bad_report_type_raises():
    with pytest.raises(TypeError, match=r"must be a dict or expose a kpis\(\) method"):
        aquakin.kpi_comparison({"a": 123})


# design -----------------------------------------------------------------------


def test_optimize_design_input_names_length_mismatch_raises():
    with pytest.raises(ValueError, match="input_names has 1 entries but bounds has d=2"):
        aquakin.optimize_design(
            lambda x: x[0] + x[1], bounds=[(0.0, 1.0), (0.0, 1.0)], input_names=["only_one"]
        )
