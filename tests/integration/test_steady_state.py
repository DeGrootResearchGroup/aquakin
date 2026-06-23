"""Algebraic plant steady state via pseudo-transient continuation (PTC).

The fast tests exercise the solver core (:mod:`aquakin.plant.steady`) on a small
analytic system -- convergence to the known root and the exact
implicit-function-theorem parameter gradient. The slow tests run the full
``Plant.steady_state`` against the forward integrate-to-steady-state reference on
the BSM plants (including the stiff BSM2 digester).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.steady import ptc_forward, solve_steady_state


# --- solver core, no plant (fast) --------------------------------------------

def test_ptc_converges_to_linear_root():
    # dy/dt = -k*y + b has the steady state y* = b/k. PTC must find it from a
    # cold start across a stiff range of rate constants (here ~5 orders, the
    # span a real plant shows between fast aeration and slow biomass).
    k = jnp.array([1.0e-2, 1.0, 50.0, 1.0e3])
    y_target = jnp.array([12.0, 8.0, 4.0, 1.5])    # comparable magnitudes
    b = k * y_target
    def rhs(y, p):
        return -k * y + p
    y0 = jnp.ones_like(b)
    res = solve_steady_state(rhs, b, y0, tol=1e-10)
    assert bool(res.converged)
    np.testing.assert_allclose(np.asarray(res.state), np.asarray(y_target), rtol=1e-6)


def test_ptc_nonnegativity():
    # With nonneg=True the iterate never leaves the physical cone even when the
    # transient would dip a species below zero.
    def rhs(y, p):
        return p - y          # steady state y* = p
    res = solve_steady_state(rhs, jnp.array([5.0, 5.0]), jnp.array([0.0, 0.0]),
                             nonneg=True, tol=1e-10)
    assert np.all(np.asarray(res.state) >= 0.0)
    np.testing.assert_allclose(np.asarray(res.state), [5.0, 5.0], rtol=1e-6)


def test_ptc_ift_gradient_matches_analytic():
    # For dy/dt = -k*y + p, y* = p/k, so dy*_i/dp_j = delta_ij / k_i. The IFT
    # gradient the solver attaches must reproduce this exactly.
    k = jnp.array([0.5, 2.0, 10.0])
    def rhs(y, p):
        return -k * y + p
    y0 = jnp.zeros(3)
    p = jnp.array([1.0, 4.0, 7.0])
    # d(sum y*)/dp_j = 1/k_j
    g = jax.grad(lambda pp: jnp.sum(solve_steady_state(rhs, pp, y0, tol=1e-11).state))(p)
    np.testing.assert_allclose(np.asarray(g), np.asarray(1.0 / k), rtol=1e-6)
    # full Jacobian dy*/dp = diag(1/k)
    J = jax.jacrev(lambda pp: solve_steady_state(rhs, pp, y0, tol=1e-11).state)(p)
    np.testing.assert_allclose(np.asarray(J), np.diag(np.asarray(1.0 / k)), atol=1e-6)


def test_ptc_reports_non_convergence():
    # A one-iteration budget cannot converge; the flag says so (and the eager
    # diagnostics are concrete).
    def rhs(y, p):
        return -y + p
    res = solve_steady_state(rhs, jnp.array([3.0]), jnp.array([0.0]),
                             max_iter=1, tol=1e-12)
    assert not bool(res.converged)
    assert int(res.iterations) == 1


def test_ptc_step_guard_keeps_overshoot_finite():
    # A large initial pseudo-timestep makes the first Newton step overshoot into a
    # region where the field is non-finite (exp blows up). The accept-always
    # iteration would propagate the resulting NaN; the step-acceptance guard
    # rejects the step, hard-shrinks dt, and still converges to the root y*=log(p).
    def rhs(y, p):
        return p - jnp.exp(y)
    res = solve_steady_state(rhs, jnp.array([1.0]), jnp.array([-50.0]),
                             dt0=1e4, scale_floor=1.0, nonneg=False, tol=1e-10)
    assert bool(jnp.all(jnp.isfinite(res.state)))
    assert bool(res.converged)
    np.testing.assert_allclose(np.asarray(res.state), [0.0], atol=1e-6)  # log(1)=0


# --- full plant: BSM1 / BSM2 (slow) ------------------------------------------

def _bsm1():
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start
    from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
    from aquakin.plant.influent import InfluentSeries
    asm1 = aquakin.load_network("asm1")
    C0 = asm1.concentrations({
        "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
        "XB_A": 0.0, "XP": 0.0, "SO": 0.0, "SNO": 0.0, "SNH": 31.56,
        "SND": 6.95, "XND": 10.59, "SALK": 7.0})
    feed = InfluentSeries(t=jnp.asarray([0.0, 100.0]), Q=jnp.full((2,), BSM1_Q_AVG),
                          C=jnp.tile(C0, (2, 1)), network=asm1)
    plant = build_bsm1(network=asm1)
    plant.add_influent("feed", feed, to="inlet_mix.fresh")
    return plant, asm1, bsm1_warm_start(plant)


@pytest.fixture(scope="module")
def bsm1():
    """The BSM1 plant built once and shared across the steady-state tests that
    only READ it -- ``steady_state`` / ``run_to_steady_state`` / ``jax.grad`` do
    not mutate the plant, so its per-instance compiled-solve cache amortises the
    PTC compile across these tests instead of rebuilding + recompiling per test.
    (The cache-assertion test below builds its own fresh plant: it asserts an
    empty cache.)"""
    return _bsm1()


@pytest.mark.slow
def test_bsm1_steady_state_matches_forward(bsm1):
    plant, asm1, y0 = bsm1
    ss = plant.steady_state(y0=y0)
    assert ss.method == "ptc" and bool(ss.converged)
    assert float(ss.residual) < 1e-5
    fwd = plant.run_to_steady_state(y0=y0, max_time=300.0)
    # The two independent steady-state routes agree on the operating point.
    i = asm1.species_index
    a = plant.states_by_unit(ss.state)
    b = plant.states_by_unit(fwd.state)
    for sp in ["XB_H", "XB_A", "SNH", "SNO", "SO"]:
        assert abs(float(a["tank5"][i[sp]]) - float(b["tank5"][i[sp]])) <= \
            0.03 * abs(float(b["tank5"][i[sp]])) + 0.05, sp


@pytest.mark.slow
def test_bsm1_steady_state_differentiable(bsm1):
    # The steady state carries the IFT parameter gradient; check it against a
    # central finite difference in the heterotroph max-growth rate.
    plant, asm1, y0 = bsm1
    start, _ = plant._state_layout["tank5"]
    idx = start + asm1.species_index["XB_H"]
    params = plant.default_parameters()

    def loss(p):
        return plant.steady_state(p, y0=y0, tol=1e-9).state[idx]

    g = jax.grad(loss)(params)
    assert bool(jnp.all(jnp.isfinite(g)))
    k = plant._parameter_layout.network_param_blocks["asm1"][0] + asm1.param_index["muH"]
    h = 1e-3 * float(params[k])
    fd = (float(loss(params.at[k].add(h))) - float(loss(params.at[k].add(-h)))) / (2 * h)
    assert abs(float(g[k]) - fd) <= 1e-4 * abs(fd) + 1e-6


@pytest.mark.slow
def test_bsm1_steady_state_differentiable_wrt_influent_load(bsm1):
    # A design sweep: the steady state is differentiable w.r.t. the influent load
    # (passed via design={"influent": ...}), not just the kinetic parameters.
    # Check d(effluent ammonia)/d(influent ammonia) against a finite difference.
    plant, asm1, y0 = bsm1
    from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
    start, _ = plant._state_layout["tank5"]
    eff = start + asm1.species_index["SNH"]
    j = asm1.species_index["SNH"]
    C_in = asm1.concentrations({
        "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
        "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0})

    def eff_snh(influent_snh):
        C = C_in.at[j].set(influent_snh)
        design = {"influent": {"feed": {"Q": jnp.asarray(BSM1_Q_AVG), "C": C}}}
        return plant.steady_state(y0=y0, design=design, tol=1e-9).state[eff]

    x0 = float(C_in[j])
    g = jax.grad(eff_snh)(x0)
    assert np.isfinite(float(g)) and float(g) > 0.0   # more load -> more residual
    h = 1e-2 * x0
    fd = (float(eff_snh(x0 + h)) - float(eff_snh(x0 - h))) / (2 * h)
    assert abs(float(g) - fd) <= 1e-2 * abs(fd) + 1e-5


@pytest.mark.slow
def test_bsm1_steady_state_differentiable_wrt_recycle_flow(bsm1):
    # The SRT / recycle design knob: the steady state is differentiable w.r.t. a
    # flow setpoint (the RAS pump flow), now a first-class plant parameter
    # addressed "<unit>.<setpoint>". Check d(effluent ammonia)/d(RAS flow) vs FD.
    plant, asm1, y0 = bsm1
    p = plant.default_parameters()
    ras = plant.parameter_index("underflow_split.ras")
    assert "clarifier.underflow_Q" in plant.parameter_names()   # clarifier knob too
    start, _ = plant._state_layout["tank5"]
    eff = start + asm1.species_index["SNH"]

    def eff_snh(params):
        return plant.steady_state(params, y0=y0, tol=1e-9).state[eff]

    g = jax.grad(eff_snh)(p)
    assert bool(jnp.all(jnp.isfinite(g)))
    h = 1e-3 * float(p[ras])
    fd = (float(eff_snh(p.at[ras].add(h))) - float(eff_snh(p.at[ras].add(-h)))) / (2 * h)
    assert abs(float(g[ras]) - fd) <= 1e-2 * abs(fd) + 1e-6
    # more RAS recycle retains more biomass -> lower effluent ammonia
    assert float(g[ras]) < 0.0


@pytest.mark.slow
def test_bsm1_steady_state_solve_is_cached():
    # The eager PTC while_loop recompiles on every call; a persisted jitted solver
    # makes a repeated concrete steady_state reuse the compile. Pin the behaviour
    # (not the wall time): one cache entry, reused across params, bit-identical
    # re-call, and the gradient path bypasses the cache (and stays finite).
    plant, _asm1, y0 = _bsm1()
    params = plant.default_parameters()
    assert not plant._steady_jit_cache                 # empty before first solve
    r1 = plant.steady_state(params, y0=y0)
    assert bool(r1.converged)
    assert len(plant._steady_jit_cache) == 1           # compiled + cached once
    # A swept-params call reuses the SAME compiled solver (rhs reads params as an
    # argument), so no new entry is added.
    r2 = plant.steady_state(params.at[0].multiply(1.001), y0=y0)
    assert bool(r2.converged) and len(plant._steady_jit_cache) == 1
    # Same-params re-call is bit-identical (the cached compiled solve).
    r3 = plant.steady_state(params, y0=y0)
    assert float(jnp.max(jnp.abs(r1.state - r3.state))) == 0.0
    # Under a gradient the call is traced, so it takes the IFT path, NOT the
    # concrete cache -- no concrete entry is added and the gradient is finite.
    g = jax.grad(lambda p: plant.steady_state(p, y0=y0).state.sum())(params)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert len(plant._steady_jit_cache) == 1


@pytest.mark.slow
def test_bsm1_steady_state_falls_back_to_forward(bsm1):
    # If PTC is starved of iterations it falls back to the forward solve.
    plant, _asm1, y0 = bsm1
    ss = plant.steady_state(y0=y0, max_iter=1, fallback=True,
                            fallback_kwargs={"max_time": 300.0})
    assert ss.method == "ptc->forward"
    assert bool(ss.converged)


@pytest.mark.slow
def test_bsm2_steady_state_matches_forward():
    # BSM2 -- the 167-state plant with the long-SRT anaerobic digester, the stiff
    # case a plain Newton root-find stalls on. PTC reaches it and agrees with the
    # forward reference.
    from aquakin.plant.bsm.bsm2 import (
        build_bsm2, bsm2_constant_influent, bsm2_parameters)
    from aquakin.plant.bsm import bsm2_warm_start
    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    y0 = bsm2_warm_start(plant)
    params = bsm2_parameters(asm1, adm1)
    ss = plant.steady_state(params, y0=y0)
    assert ss.method == "ptc" and bool(ss.converged)
    assert float(ss.residual) < 1e-5
    fwd = plant.run_to_steady_state(params, y0=y0, max_time=400.0, max_steps=800_000)
    i = asm1.species_index
    a = plant.states_by_unit(ss.state)
    b = plant.states_by_unit(fwd.state)
    for sp in ["XB_H", "XB_A", "SNH", "SNO"]:
        assert abs(float(a["tank5"][i[sp]]) - float(b["tank5"][i[sp]])) <= \
            0.03 * abs(float(b["tank5"][i[sp]])) + 0.05, sp


def test_ptc_step_guard_rejects_finite_blowup():
    # The growth guard must reject a step whose residual blows up by a large but
    # *finite* factor (a Newton overshoot from a flat region), not only a
    # non-finite one. Cubic dy/dt = p - y^3 (root y*=1): from y0=0.01 the Jacobian
    # -3y^2 is ~0, so a large dt0 makes ONE step overshoot to y~3e3 where the
    # residual is ~1e7x the start. With the default divergence_factor the step is
    # rejected (the iterate is held) and dt hard-shrunk; with divergence_factor=inf
    # (reject only non-finite) the same finite blow-up is accepted and the iterate
    # jumps. Deterministic: it checks the acceptance logic, not a convergence count.
    def rhs(y, p):
        return p - y ** 3
    y0 = jnp.array([0.01])
    p = jnp.array([1.0])
    kw = dict(dt0=1e6, scale_floor=1.0, nonneg=False)
    held, *_ = ptc_forward(rhs, p, y0, max_iter=1, divergence_factor=1000.0, **kw)
    jumped, *_ = ptc_forward(rhs, p, y0, max_iter=1, divergence_factor=jnp.inf, **kw)
    assert float(held[0]) == pytest.approx(0.01, abs=1e-6)   # blow-up rejected
    assert float(jumped[0]) > 1e3                            # blow-up accepted
    # The rejection does not break convergence: the guarded solve still finds y*=1.
    res = solve_steady_state(rhs, p, y0, dt0=1e6, scale_floor=1.0, nonneg=False,
                             tol=1e-10)
    assert bool(res.converged)
    np.testing.assert_allclose(np.asarray(res.state), [1.0], atol=1e-6)


@pytest.mark.slow
def test_bsm2_steady_state_per_state_scaling_cuts_iterations():
    # The default per-state pseudo-time / residual floor (max(|y0|, 1e-6)) gives
    # every state a magnitude-consistent scale, so the SER ramp is no longer
    # throttled by the over-damped small-magnitude states -- roughly halving the
    # PTC iteration count vs the old flat scalar floor, while converging to the
    # same root.
    from aquakin.plant.bsm.bsm2 import (
        build_bsm2, bsm2_constant_influent, bsm2_parameters)
    from aquakin.plant.bsm import bsm2_warm_start
    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    y0 = bsm2_warm_start(plant)
    params = bsm2_parameters(asm1, adm1)

    default = plant.steady_state(params, y0=y0)                # per-state floor
    flat = plant.steady_state(params, y0=y0, scale_floor=1.0)  # old behaviour
    assert bool(default.converged) and bool(flat.converged)
    # Fewer iterations (the win) ...
    assert int(default.iterations) < int(flat.iterations)
    # ... and the same operating point (scaling changes the path, not the root).
    rel = float(jnp.max(jnp.abs(default.state - flat.state)
                        / (jnp.abs(flat.state) + 1e-9)))
    assert rel < 1e-4


@pytest.mark.slow
def test_steady_state_forward_mode_ad():
    """The steady-state IFT gradient is now a ``custom_jvp``, so the plant steady
    state is differentiable in BOTH directions: forward-mode ``jacfwd`` flows
    through ``plant.steady_state`` (previously rejected by the reverse-only
    ``custom_vjp``), and it agrees with reverse-mode ``jax.grad`` and with finite
    differences. This is the forward-mode capability the many-output sensitivity
    screen needs."""
    plant, asm1, y0 = _bsm1()
    base = plant.default_parameters()
    i = plant.parameter_index("asm1.muA")
    si = asm1.species_index

    def out(theta):
        s = plant.steady_state(base.at[i].set(theta), y0=y0).state
        return plant.states_by_unit(s)["tank5"][si["SNO"]]

    th = float(base[i])
    fwd = float(jax.jacfwd(out)(th))                 # forward mode (the new path)
    rev = float(jax.grad(out)(th))                   # reverse mode (unchanged)
    h = th * 1e-4
    fd = (float(out(th + h)) - float(out(th - h))) / (2.0 * h)
    assert np.isfinite(fwd) and np.isfinite(rev)
    assert fwd == pytest.approx(rev, rel=1e-7)       # both IFT, same root
    assert fwd == pytest.approx(fd, rel=1e-4)        # the true sensitivity (FD floor)


@pytest.mark.slow
def test_steady_state_sensitivity_helper():
    """``plant.steady_state_sensitivity`` returns the exact IFT output sensitivity
    in either AD direction from a single steady-state solve. Forward and reverse
    give the same result; it matches a ``jax.grad`` through ``steady_state`` and
    finite differences; the elasticity option is finite."""
    plant, asm1, y0 = _bsm1()
    base = plant.default_parameters()
    si = asm1.species_index

    def out_fn(y):
        sb = plant.states_by_unit(y)
        return jnp.array([sb["tank5"][si["SNH"]], sb["tank5"][si["SNO"]]])

    Sf = np.asarray(plant.steady_state_sensitivity(
        base, y0=y0, output_fn=out_fn, mode="forward"))
    Sr = np.asarray(plant.steady_state_sensitivity(
        base, y0=y0, output_fn=out_fn, mode="reverse"))
    assert Sf.shape == (2, base.shape[0])
    # forward and reverse are the same exact sensitivity
    assert np.allclose(Sf, Sr, rtol=1e-7, atol=1e-12)

    # matches a gradient through the solve, and finite differences
    i = plant.parameter_index("asm1.muA")
    th = float(base[i])

    def scalar(theta):
        s = plant.steady_state(base.at[i].set(theta), y0=y0).state
        return plant.states_by_unit(s)["tank5"][si["SNO"]]

    g = float(jax.grad(scalar)(th))
    h = th * 1e-4
    fd = (scalar(th + h) - scalar(th - h)) / (2.0 * h)
    assert float(Sf[1, i]) == pytest.approx(g, rel=1e-7)
    assert float(Sf[1, i]) == pytest.approx(float(fd), rel=1e-4)  # FD floor

    # a parameter subset (wrt) equals the full computation's selected columns
    wrt = ["asm1.muH", "asm1.muA", "asm1.etag"]
    widx = [plant.parameter_index(w) for w in wrt]
    Sw = np.asarray(plant.steady_state_sensitivity(
        base, y0=y0, output_fn=out_fn, wrt=wrt, mode="forward"))
    assert Sw.shape == (2, len(wrt))
    assert np.allclose(Sw, Sf[:, widx], rtol=1e-7, atol=1e-12)

    # state= (a pre-solved steady state) skips the internal solve, same result
    ss = plant.steady_state(base, y0=y0).state
    Ss = np.asarray(plant.steady_state_sensitivity(
        base, state=ss, output_fn=out_fn, mode="forward"))
    assert np.allclose(Ss, Sf, rtol=1e-7, atol=1e-12)

    E = plant.steady_state_sensitivity(base, y0=y0, output_fn=out_fn,
                                       elasticity=True)
    assert bool(jnp.all(jnp.isfinite(E)))


@pytest.mark.slow
def test_steady_state_dgsm_matches_dgsm():
    """plant.steady_state_dgsm screens the steady state globally (DGSM) by reusing
    the IFT helper per sample. With the same Sobol seed it draws the same points
    and applies the same Sobol total-index formula as the generic aquakin.dgsm, so
    the bounds are identical -- just computed more cheaply (one dF/dy factorisation
    per sample). The convergence() sample-size study runs from the retained
    per-sample data."""
    plant, asm1, y0 = _bsm1()
    base = plant.default_parameters()
    si = asm1.species_index
    screen = ["asm1.muH", "asm1.muA", "asm1.etag", "asm1.bA"]
    idx = [plant.parameter_index(s) for s in screen]
    val = np.array([float(base[i]) for i in idx])
    ranges = np.array([[v * 0.75, v * 1.25] for v in val])

    def out_fn(y):
        sb = plant.states_by_unit(y)
        return jnp.array([sb["tank5"][si["SNH"]], sb["tank5"][si["SNO"]]])

    res = plant.steady_state_dgsm(
        ranges, output_fn=out_fn, output_names=["SNH", "SNO"],
        wrt=screen, n_samples=32, seed=0, y0=y0)
    assert res.sobol_total_bound.shape == (2, 4)
    assert res.n_samples == 32

    # same Sobol points + DGSM formula as aquakin.dgsm => identical bounds
    def fn_snh(x):
        pp = base.at[jnp.asarray(idx)].set(jnp.asarray(x))
        s = plant.steady_state(pp, y0=y0).state
        return plant.states_by_unit(s)["tank5"][si["SNH"]]

    d = aquakin.dgsm(fn_snh, ranges, input_names=screen, n_samples=32, seed=0,
                     ad_mode="reverse")
    mine, theirs = dict(res.ranked("SNH")), dict(d.ranked())
    for s in screen:
        assert mine[s] == pytest.approx(theirs[s], rel=1e-6, abs=1e-12)

    # convergence (sample-size study): running bound + MC std error vs sample count
    counts, bound, se = res.convergence()
    assert bound.shape == (len(counts), 2, 4)
    assert se.shape == bound.shape
    j = res.input_names.index("asm1.muA")               # dominant input
    assert float(se[-1, 0, j]) < float(se[0, 0, j])     # std error shrinks with N


def test_dgsm_cond_filter_and_with_cond_factor():
    """The near-singular-Jacobian filter (the heavy-tail robustification) drops the
    samples it should and re-aggregation reuses the retained per-sample data. Fast
    and deterministic -- exercises the aggregation directly, no plant solve."""
    from aquakin.plant.plant import (_dgsm_aggregate, _cond_mask,
                                     SteadyStateDGSMResult)
    rng = np.random.default_rng(0)
    N, m, k = 200, 1, 3
    grad = rng.normal(size=(N, m, k))
    grad[0] = 1.0e6                       # one near-singular sample: huge sensitivity
    grad_sq = grad ** 2
    outputs = rng.normal(size=(N, m))
    cond = np.full(N, 1.0e10)
    cond[0] = 1.0e20                      # ... at a near-singular Jacobian
    rng2 = np.ones(k)

    mask = _cond_mask(cond, cond_factor=100.0)
    assert not bool(mask[0]) and bool(mask[1:].all())
    assert bool(_cond_mask(cond, None).all())          # None keeps everything

    b_filt, *_ = _dgsm_aggregate(grad_sq, outputs, rng2, sample_mask=mask)
    b_all, *_ = _dgsm_aggregate(grad_sq, outputs, rng2)   # includes the huge sample
    assert np.all(b_filt < b_all)                      # the outlier inflates the mean

    res = SteadyStateDGSMResult(
        input_names=["a", "b", "c"], output_names=["o"],
        sobol_total_bound=jnp.asarray(b_all), std_error=jnp.zeros((m, k)),
        nu=jnp.zeros((m, k)), output_variance=jnp.zeros(m),
        ranges=jnp.asarray(np.tile([0.0, 1.0], (k, 1))),
        n_samples=N, seed=0, grad_sq=jnp.asarray(grad_sq),
        outputs=jnp.asarray(outputs), n_valid=jnp.full(m, N),
        cond=jnp.asarray(cond), cond_factor=None)
    res2 = res.with_cond_factor(100.0)                 # re-aggregate, no re-solve
    assert int(res2.n_valid[0]) == N - 1
    np.testing.assert_allclose(np.asarray(res2.sobol_total_bound),
                               np.asarray(b_filt), rtol=1e-12)
