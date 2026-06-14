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
        _fn, {"a": (0.0, 1.0), "b": (2.0, 4.0)},
        output_names=["sum", "prod"], n_samples=64, seed=0)
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
    mc = aquakin.monte_carlo(_fn, [(0.0, 1.0), (0.0, 1.0)],
                             n_samples=64, sampler=sampler, seed=1)
    assert mc.sampler == sampler
    assert mc.n_valid > 0
    assert np.all(np.isfinite(mc.outputs))


def test_monte_carlo_recovers_input_distribution_moments():
    # fn returns the inputs unchanged, so the sampled output marginals must match
    # the requested distributions' moments (a check that inverse-transform
    # sampling of normal / lognormal is wired correctly).
    mc = aquakin.monte_carlo(
        lambda x: x,
        {"normal": {"dist": "normal", "mean": 5.0, "std": 1.0},
         "lognorm": {"dist": "lognormal", "mean": 2.0, "std": 0.5}},
        output_names=["normal", "lognorm"], n_samples=4096, seed=0)
    norm_col = mc.output_named("normal")
    ln_col = mc.output_named("lognorm")
    assert norm_col.mean() == pytest.approx(5.0, abs=0.1)
    assert norm_col.std() == pytest.approx(1.0, abs=0.1)
    assert ln_col.mean() == pytest.approx(2.0, rel=0.1)
    assert ln_col.std() == pytest.approx(0.5, rel=0.2)
    assert np.all(ln_col > 0.0)                       # lognormal is positive


def test_monte_carlo_scalar_output_and_dropping_nonfinite():
    # fn returns inf when a > 0.9, so those rows are dropped.
    def fn(x):
        return jnp.where(x[0] > 0.9, jnp.inf, x[0] + x[1])
    mc = aquakin.monte_carlo(fn, [(0.0, 1.0), (0.0, 1.0)], n_samples=128, seed=0)
    assert mc.output_names == ["output"]
    assert mc.n_valid < mc.n_drawn                    # some rows dropped
    assert np.all(np.isfinite(mc.outputs))


def test_monte_carlo_summary_is_a_table():
    mc = aquakin.monte_carlo(_fn, [(0.0, 1.0), (0.0, 1.0)],
                             output_names=["sum", "prod"], n_samples=16)
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
        input_names=["a", "b"], baseline=[0.5, 2.0],
        output_names=["sum", "prod"])
    assert sc.scenario_names == ["base", "hi_a", "hi_b"]
    # base: 0.5+2=2.5 ; hi_a: 1+2=3 ; hi_b: 0.5+10=10.5
    assert np.allclose(sc.output_named("sum"), [2.5, 3.0, 10.5])
    assert sc.best("sum", minimize=True) == "base"
    assert sc.best("sum", minimize=False) == "hi_b"


def test_compare_scenarios_full_vectors_and_table():
    sc = aquakin.compare_scenarios(
        _fn, {"x": [1.0, 1.0], "y": [2.0, 3.0]},
        input_names=["a", "b"], output_names=["sum", "prod"])
    assert np.allclose(sc.output_named("prod"), [1.0, 6.0])
    assert "scenario" in sc.table() and "x" in sc.table()


def test_compare_scenarios_unknown_override_raises():
    with pytest.raises(KeyError, match="unknown input"):
        aquakin.compare_scenarios(_fn, {"s": {"ghost": 1.0}},
                                  input_names=["a", "b"], baseline=[0.0, 0.0])


def test_compare_scenarios_nonfinite_raises():
    def fn(x):
        return jnp.where(x[0] > 5.0, jnp.nan, x[0])
    with pytest.raises(ValueError, match="non-finite"):
        aquakin.compare_scenarios(fn, {"ok": {"a": 1.0}, "bad": {"a": 9.0}},
                                  input_names=["a", "b"], baseline=[0.0, 0.0])


# --- end-to-end with a real solve --------------------------------------------

def test_monte_carlo_through_a_reactor_solve():
    # The full path: fn builds params and runs a stiff ASM solve. Uncertain AOB
    # max growth rate -> effluent ammonia distribution.
    net = aquakin.load_network("asm3_2step")
    r = aquakin.BatchReactor(net, aquakin.SpatialConditions.uniform(T=293.15))
    C0 = net.concentrations({"SO2": 300.0, "SNH4": 30.0, "XAOB": 80.0,
                             "XNOB": 80.0, "SALK": 0.05})
    i_mu = net.param_index["muAOB"]

    def fn(x):
        p = net.default_parameters().at[i_mu].set(x[0])
        sol = r.solve(C0, params=p, t_span=(0.0, 1.0),
                      t_eval=jnp.linspace(0.0, 1.0, 6))
        return sol.C_named("SNH4")[-1]

    mc = aquakin.monte_carlo(fn, {"muAOB": (0.6, 1.2)}, n_samples=16, seed=0)
    assert mc.n_valid == 16
    assert np.all(np.isfinite(mc.outputs))
    # Faster AOB -> lower effluent ammonia, so the output spans a real range.
    assert mc.std()[0] > 0.0
