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


def test_arclength_solves_reachable_and_detects_past_fold():
    # Saddle-node test: F(y, lambda) = lambda - y^2, steady state y = sqrt(lambda),
    # with a fold at lambda = 0 (the upper branch y=+sqrt and the lower y=-sqrt
    # meet and annihilate). Continue from the known (lambda=1, y=1).
    from aquakin.plant.steady import arclength_continuation_solve

    def rhs(y, p):
        return jnp.array([p[0] - y[0] ** 2])

    pk, yk = jnp.array([1.0]), jnp.array([1.0])
    kw = dict(scale=jnp.array([1.0]), ptc_kwargs=dict(scale_floor=jnp.array([1.0])))

    # Reachable: lambda 1 -> 4 stays on the branch; the augmented corrector tracks
    # it to y = 2 (no fold crossed).
    r_ok = arclength_continuation_solve(rhs, pk, yk, jnp.array([4.0]), **kw)
    assert r_ok.status == "converged"
    np.testing.assert_allclose(np.asarray(r_ok.state), [2.0], atol=1e-4)

    # Past the fold: lambda 1 -> -1 drives the parameter through the fold at
    # lambda=0 (arc s=0.5), where the operating branch ceases to exist; the
    # tangent's s-velocity reverses and the solve reports it does not exist.
    r_fold = arclength_continuation_solve(rhs, pk, yk, jnp.array([-1.0]), **kw)
    assert r_fold.status == "past_fold"
    assert float(r_fold.s_max) < 1.0
    # The fold of this system is at s=0.5 (lambda crosses 0 halfway).
    assert 0.4 < float(r_fold.s_max) < 0.65
    # Default records no branch.
    assert r_fold.branch is None


def test_arclength_record_branch_traces_the_curve():
    """``record_branch=True`` returns the traced ``(s, y)`` path and still reports
    the fold; every recorded point lies on the branch ``y**2 = p``."""
    from aquakin.plant.steady import arclength_continuation_solve

    def rhs(y, p):
        return jnp.array([p[0] - y[0] ** 2])

    pk, yk = jnp.array([1.0]), jnp.array([1.0])
    kw = dict(scale=jnp.array([1.0]), ptc_kwargs=dict(scale_floor=jnp.array([1.0])))
    r = arclength_continuation_solve(rhs, pk, yk, jnp.array([-1.0]),
                                     record_branch=True, max_steps=300, **kw)
    assert r.status == "past_fold" and 0.4 < float(r.s_max) < 0.65
    s, ys = r.branch
    assert s.shape[0] == ys.shape[0] >= 3
    p = 1.0 - 2.0 * np.asarray(s)                    # p = pk + s*(target - pk)
    y = np.asarray(ys)[:, 0]
    np.testing.assert_allclose(y ** 2, p, atol=1e-4)  # each point solves F=0
    assert float(y[0]) == pytest.approx(1.0, abs=1e-3) and p.min() < 0.05  # traced to the nose


# --- natural-parameter continuation (fast) -----------------------------------

def test_continuation_solve_reaches_far_root():
    # Natural-parameter predictor-corrector continuation from a known steady
    # state to a far parameter set. For the nonlinear dy/dt = p - y**3 the branch
    # is y*(p) = p**(1/3), so the tangent predictor is only first-order and the PTC
    # corrector does genuine work at each step. Exercises the internally-built
    # (predict, correct) kernels (kernels=None branch) end to end.
    from aquakin.plant.steady import continuation_solve

    def rhs(y, p):
        return p - y ** 3
    pk = jnp.array([1.0, 8.0])
    yk = pk ** (1.0 / 3.0)                        # a known steady state (rhs ~ 0)
    pt = jnp.array([27.0, 64.0])                  # far target -> y* = [3.0, 4.0]
    res = continuation_solve(rhs, pk, yk, pt, tol=1e-10)
    assert bool(res.converged)
    np.testing.assert_allclose(np.asarray(res.state), [3.0, 4.0], rtol=1e-6)
    # It took real work along the path and the (inexact) predictor made the
    # corrector iterate.
    assert res.continuation_steps > 0
    assert res.corrector_iterations > 0


def test_continuation_solve_reuses_prebuilt_kernels():
    # make_continuation_kernels builds the jitted (predict, correct) once so a
    # sweep over several targets reuses one compilation. Pass them via kernels=
    # (the kernels-provided branch) and check two different targets both land on
    # their analytic roots with the same kernels.
    from aquakin.plant.steady import continuation_solve, make_continuation_kernels

    k = jnp.array([2.0, 0.5])
    def rhs(y, p):
        return -k * y + p
    pk = jnp.array([2.0, 0.5])
    yk = pk / k                                  # = [1.0, 1.0]
    kernels = make_continuation_kernels(rhs)
    assert len(kernels) == 2                     # (predict, correct)

    for pt in (jnp.array([8.0, 2.0]), jnp.array([4.0, 4.0])):
        res = continuation_solve(rhs, pk, yk, pt, kernels=kernels, tol=1e-10)
        assert bool(res.converged)
        np.testing.assert_allclose(np.asarray(res.state), np.asarray(pt / k), rtol=1e-6)


def test_continuation_solve_reports_incomplete_path():
    # A one-step budget cannot span s: 0 -> 1 in a single ds0=0.25 move, so the
    # path stops short and reports non-convergence -- while still having taken and
    # accepted that one continuation step (the loop-exhaustion return).
    from aquakin.plant.steady import continuation_solve

    def rhs(y, p):
        return -y + p                            # y* = p, easy corrector
    pk, yk = jnp.array([1.0]), jnp.array([1.0])
    pt = jnp.array([5.0])
    res = continuation_solve(rhs, pk, yk, pt, ds0=0.25, max_continuation_steps=1)
    assert not bool(res.converged)
    assert res.continuation_steps == 1           # one step accepted, then budget spent


def test_continuation_solve_shrinks_and_stops_on_corrector_failure():
    # When every corrector fails (here starved to a single PTC iteration against a
    # tight tolerance on a nonlinear branch), the adaptive step ds is shrunk on
    # each failure and the solve stops once ds falls below ds_min -- reporting
    # non-convergence with no step accepted. Exercises the corrector-failure
    # branch of the continuation loop.
    from aquakin.plant.steady import continuation_solve

    def rhs(y, p):
        return p - y ** 3
    pk = jnp.array([1.0])
    yk = pk ** (1.0 / 3.0)
    pt = jnp.array([27.0])                        # far enough that one PTC step misses
    res = continuation_solve(rhs, pk, yk, pt, ds0=0.25, ds_min=1e-3,
                             max_continuation_steps=200, max_iter=1, tol=1e-12)
    assert not bool(res.converged)
    assert res.continuation_steps == 0           # no step ever accepted


# --- pseudo-arclength continuation: auto-scale and prebuilt kernels (fast) ----

def test_arclength_auto_scale_reaches_root():
    # With no scale= given, the solver auto-scales by max(|y_known|, 1e-3). On the
    # saddle-node F = p - y^2 the reachable leg (lambda 1 -> 4) must still track the
    # branch to y = 2. Covers the scale-None default and the converged property.
    from aquakin.plant.steady import arclength_continuation_solve

    def rhs(y, p):
        return jnp.array([p[0] - y[0] ** 2])
    pk, yk = jnp.array([1.0]), jnp.array([1.0])
    r = arclength_continuation_solve(rhs, pk, yk, jnp.array([4.0]))   # scale=None
    assert r.status == "converged"
    assert bool(r.converged)                     # the converged property
    np.testing.assert_allclose(np.asarray(r.state), [2.0], atol=1e-4)


def test_arclength_reuses_prebuilt_kernels():
    # make_arclength_kernels builds the jitted (corrector, tangent, init_tangent)
    # for a fixed scale; pass them via kernels= (the kernels-provided branch) with
    # the matching scale and confirm the reachable leg still converges to y = 2.
    from aquakin.plant.steady import arclength_continuation_solve, make_arclength_kernels

    def rhs(y, p):
        return jnp.array([p[0] - y[0] ** 2])
    scale = jnp.array([1.0])
    ptc_kwargs = dict(scale_floor=jnp.array([1.0]))
    kernels = make_arclength_kernels(rhs, scale, ptc_kwargs)
    assert len(kernels) == 3                     # (corrector, tangent, init_tangent)

    pk, yk = jnp.array([1.0]), jnp.array([1.0])
    r = arclength_continuation_solve(rhs, pk, yk, jnp.array([4.0]),
                                     kernels=kernels, scale=scale,
                                     ptc_kwargs=ptc_kwargs)
    assert r.status == "converged"
    np.testing.assert_allclose(np.asarray(r.state), [2.0], atol=1e-4)


def test_branch_recorder_enabled_and_disabled():
    """The branch-recording bookkeeping factored out of the arclength loop: when
    disabled every method is a no-op and result() is None; when enabled it seeds
    with the known start and accumulates (s, y) points into (s, ys) arrays."""
    from aquakin.plant.steady import _BranchRecorder

    off = _BranchRecorder(False, jnp.array([1.0, 2.0]))
    off.record(0.5, jnp.array([9.0, 9.0]))       # no-op
    assert off.result() is None

    on = _BranchRecorder(True, jnp.array([1.0, 2.0]))
    on.record(0.5, jnp.array([1.5, 2.5]))
    on.record(1.0, jnp.array([2.0, 3.0]))
    s, ys = on.result()
    np.testing.assert_allclose(np.asarray(s), [0.0, 0.5, 1.0])   # seeded with s=0
    assert ys.shape == (3, 2)
    np.testing.assert_allclose(np.asarray(ys[0]), [1.0, 2.0])    # the known start
    np.testing.assert_allclose(np.asarray(ys[-1]), [2.0, 3.0])


# --- full plant: BSM1 / BSM2 (slow) ------------------------------------------

def _bsm1():
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start
    from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
    from aquakin.plant.influent import InfluentSeries
    asm1 = aquakin.load_model("asm1")
    C0 = asm1.concentrations({
        "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
        "XB_A": 0.0, "XP": 0.0, "SO": 0.0, "SNO": 0.0, "SNH": 31.56,
        "SND": 6.95, "XND": 10.59, "SALK": 7.0})
    feed = InfluentSeries(t=jnp.asarray([0.0, 100.0]), Q=jnp.full((2,), BSM1_Q_AVG),
                          C=jnp.tile(C0, (2, 1)), model=asm1)
    plant = build_bsm1(model=asm1)
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
    k = plant._parameter_layout.model_param_blocks["asm1"][0] + asm1.param_index["muH"]
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


def test_bsm1_steady_state_keeps_ptc_result_when_ladder_declines(bsm1):
    # With no fallback configured (no continuation start, arclength/forward off)
    # the escalation ladder has nothing to try, so a starved solve returns the
    # non-converged PTC result verbatim rather than a fallback method -- the
    # shared _steady_fallback_ladder "return None -> keep PTC" contract.
    plant, _asm1, y0 = bsm1
    ss = plant.steady_state(y0=y0, max_iter=1, arclength=False, fallback=False)
    assert ss.method == "ptc"
    assert not bool(ss.converged)


def _design_cstr():
    """A single CSTR (A -> B, k = 0.1, V/Q = 10) fed 1 g/m3 of A, plus a design
    override that raises the feed to 5 g/m3 (steady state A = B = 2.5)."""
    from aquakin.plant import CSTRUnit, Plant
    from aquakin.plant.influent import InfluentSeries

    net = aquakin.load_model_from_file("tests/fixtures/simple_model.yaml")
    inf = InfluentSeries(t=jnp.asarray([0.0, 100.0]), Q=jnp.asarray([10.0, 10.0]),
                         C=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]), model=net)
    plant = Plant("one")
    plant.add_unit(CSTRUnit(name="tank", model=net, volume=100.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}))
    plant.add_influent("feed", inf, to="tank.inlet")
    design = {"influent": {"feed": {"Q": jnp.asarray(10.0), "C": jnp.asarray([5.0, 0.0])}}}
    return plant, design


@pytest.mark.parametrize("extra", [{}, {"atol": 1e-10}], ids=["default_atol", "atol"])
def test_steady_state_forward_fallback_honours_design(extra):
    # A starved PTC falls back to forward integration; that fallback must
    # integrate the design-override feed, not the recorded influent (which
    # would give A = B = 0.5 and still report converged). An explicit solver
    # ``atol`` reaches the design rung too.
    plant, design = _design_cstr()
    ss = plant.steady_state(design=design, max_iter=1, tol=1e-12, arclength=False,
                            fallback_kwargs={"ss_rtol": 1e-8, "ss_atol": 1e-8, **extra})
    assert ss.method == "ptc->forward"
    assert bool(ss.converged)
    np.testing.assert_allclose(np.asarray(ss.state), [2.5, 2.5], rtol=1e-4)


@pytest.mark.parametrize("criterion", ["norm", "scaled_residual"])
def test_steady_state_design_forward_fallback_takes_criterion(criterion):
    # The design-override forward rung takes run_to_steady_state's criterion /
    # tol / scale_floor and returns the scaled residual (issue #628). With the
    # design feed equal to the recorded one it matches the plain fallback.
    plant, _ = _design_cstr()
    same = {"influent": {"feed": {"Q": jnp.asarray(10.0), "C": jnp.asarray([1.0, 0.0])}}}
    kw = {"criterion": criterion, "tol": 1e-9, "scale_floor": 1e-6,
          "ss_rtol": 1e-10, "ss_atol": 1e-10}
    common = dict(max_iter=1, tol=1e-12, arclength=False, fallback_kwargs=kw)
    ss_d = plant.steady_state(design=same, **common)
    ss_p = plant.steady_state(**common)
    for ss in (ss_d, ss_p):
        assert ss.method == "ptc->forward"
        assert bool(ss.converged)
        assert ss.residual is not None and np.isfinite(ss.residual)
    if criterion == "scaled_residual":
        assert ss_d.residual <= 1e-9
    np.testing.assert_allclose(np.asarray(ss_d.state), np.asarray(ss_p.state), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ss_d.state), [0.5, 0.5], rtol=1e-6)
    assert ss_d.residual == pytest.approx(ss_p.residual, rel=1e-3, abs=1e-9)


def test_steady_state_design_forward_fallback_rejects_bad_criterion():
    plant, design = _design_cstr()
    with pytest.raises(ValueError, match="criterion"):
        plant.steady_state(design=design, max_iter=1, tol=1e-12, arclength=False,
                           fallback_kwargs={"criterion": "bogus"})


@pytest.mark.parametrize("arclength, named", [
    (True, "continuation_from and arclength"),
    (False, "continuation_from fallback"),
], ids=["arclength", "no_arclength"])
def test_steady_state_design_skips_parameter_continuation(arclength, named):
    # The continuation rungs deform in params alone; with a design override
    # they are skipped (rather than crashing on the (params, design) pytree),
    # and the explicit request is reported rather than dropped silently.
    plant, design = _design_cstr()
    p = plant.default_parameters()
    with pytest.warns(UserWarning, match=named):
        ss = plant.steady_state(design=design, max_iter=1, tol=1e-12, fallback=False,
                                continuation_from=(p, jnp.asarray([0.5, 0.5])),
                                arclength=arclength)
    assert ss.method == "ptc"
    assert not bool(ss.converged)


def test_steady_state_design_no_skip_warning_without_continuation():
    # A caller who did not ask for continuation is not warned, even with the
    # default arclength=True (that rung needs continuation_from anyway).
    import warnings
    plant, design = _design_cstr()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r"steady_state\(design=")
        ss = plant.steady_state(design=design, max_iter=1, tol=1e-12, fallback=False)
    assert ss.method == "ptc"


@pytest.mark.slow
def test_bsm2_steady_state_matches_forward():
    # BSM2 -- the 167-state plant with the long-SRT anaerobic digester, the stiff
    # case a plain Newton root-find stalls on. PTC reaches it and agrees with the
    # forward reference.
    from aquakin.plant.bsm.bsm2 import (
        build_bsm2, bsm2_constant_influent, bsm2_parameters)
    from aquakin.plant.bsm import bsm2_warm_start
    asm1 = aquakin.load_model("asm1")
    adm1 = aquakin.load_model("adm1")
    # The default reference settler (lumped TSS + soluble holdup). Before the
    # per-state positivity limit, healthy-step dt growth and stall restart (issue
    # #627) PTC needed ~100 iterations here, erratically, and ran past 400 on
    # Linux CI; it now takes ~37, so the cap below has a wide margin.
    plant = build_bsm2(asm1_model=asm1, adm1_model=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    y0 = bsm2_warm_start(plant)
    params = bsm2_parameters(asm1, adm1)
    ss = plant.steady_state(params, y0=y0)
    assert ss.method == "ptc" and bool(ss.converged)
    assert int(ss.iterations) <= 150
    assert float(ss.residual) < 1e-5
    fwd = plant.run_to_steady_state(params, y0=y0, max_time=400.0, max_steps=800_000)
    i = asm1.species_index
    a = plant.states_by_unit(ss.state)
    b = plant.states_by_unit(fwd.state)
    for sp in ["XB_H", "XB_A", "SNH", "SNO"]:
        assert abs(float(a["tank5"][i[sp]]) - float(b["tank5"][i[sp]])) <= \
            0.03 * abs(float(b["tank5"][i[sp]])) + 0.05, sp


def test_ptc_line_search_bounds_finite_blowup():
    # A step whose residual blows up by a large but *finite* factor (a Newton
    # overshoot from a flat region) must be bounded, not only a non-finite one.
    # Cubic dy/dt = p - y^3 (root y*=1): from y0=0.01 the Jacobian -3y^2 is ~0, so a
    # large dt0 makes the full step overshoot to y~3e3 where the residual is ~1e7x
    # the start. With the default divergence_factor the backtracking line search
    # shrinks the step until its residual is within divergence_factor times the best
    # residual so far, so the iterate stays bounded (here y~26); with
    # divergence_factor=inf (no bound) the full overshoot is taken. Deterministic:
    # it checks the step-bounding logic, not a convergence count.
    def rhs(y, p):
        return p - y ** 3
    y0 = jnp.array([0.01])
    p = jnp.array([1.0])
    kw = dict(dt0=1e6, scale_floor=1.0, nonneg=False)
    bounded, *_ = ptc_forward(rhs, p, y0, max_iter=1, divergence_factor=1000.0, **kw)
    jumped, *_ = ptc_forward(rhs, p, y0, max_iter=1, divergence_factor=jnp.inf, **kw)
    assert jnp.isfinite(bounded[0])
    assert abs(float(bounded[0])) < 100.0      # overshoot backtracked to within bound
    assert float(jumped[0]) > 1e3              # full overshoot taken without the bound
    # The bounding does not break convergence: the guarded solve still finds y*=1.
    res = solve_steady_state(rhs, p, y0, dt0=1e6, scale_floor=1.0, nonneg=False,
                             tol=1e-10)
    assert bool(res.converged)
    np.testing.assert_allclose(np.asarray(res.state), [1.0], atol=1e-6)


def test_ptc_positivity_limit_is_per_state():
    # A near-zero state driven hard negative must not throttle the rest of the
    # step (issue #627). Here state 0 sits at 1e-6 with a constant sink, state 1
    # is one Newton step from its root 2. A common fraction-to-boundary step
    # length (0.95 * 1e-6 / |dy_0|) would leave state 1 essentially unmoved; the
    # per-state limit cuts only state 0, to 5% of its value, and state 1 takes
    # its full step.
    def rhs(y, p):
        return jnp.array([-1.0 + 0.0 * y[0], 2.0 - y[1]])

    y, _, k, _ = ptc_forward(rhs, None, jnp.array([1e-6, 1.0]), dt0=1e8, max_iter=1,
                             scale_floor=1.0)
    assert int(k) == 1
    np.testing.assert_allclose(float(y[0]), 0.05e-6, rtol=1e-6)
    np.testing.assert_allclose(float(y[1]), 2.0, rtol=1e-6)


def test_ptc_growth_floor_escapes_slow_plateau():
    # dy/dt = 1 - y from y=0 with a tiny dt0: every step is a clean full step but
    # the residual falls only by the fraction dt/(1+dt), so pure SER
    # (growth_floor=1) grows dt just as slowly and is still far from the root
    # at the cap. The healthy-step growth floor doubles dt per clean step.
    def rhs(y, p):
        return 1.0 - y

    kw = dict(dt0=1e-3, scale_floor=1.0, tol=1e-10)
    *_, k_ser, conv_ser = ptc_forward(rhs, None, jnp.array([0.0]), growth_floor=1.0, **kw)
    y, _, k, conv = ptc_forward(rhs, None, jnp.array([0.0]), **kw)
    assert not bool(conv_ser) and int(k_ser) == 400
    assert bool(conv) and int(k) < 30
    np.testing.assert_allclose(float(y[0]), 1.0, atol=1e-8)


@pytest.mark.parametrize("ser_norm", ["mean", "max"])
def test_ptc_ser_norm_choices_converge(ser_norm):
    def rhs(y, p):
        return 1.0 - y

    y, _, _, conv = ptc_forward(rhs, None, jnp.array([0.0]), dt0=1e-3, scale_floor=1.0,
                                tol=1e-10, ser_norm=ser_norm)
    assert bool(conv)
    np.testing.assert_allclose(float(y[0]), 1.0, atol=1e-8)


def test_ptc_rejects_unknown_ser_norm():
    with pytest.raises(ValueError, match="ser_norm"):
        ptc_forward(lambda y, p: -y, None, jnp.array([1.0]), ser_norm="l2")


def test_ptc_stall_restart_returns_to_best_iterate():
    # With the line-search bound off, the first step of the cubic (see
    # test_ptc_line_search_bounds_finite_blowup) overshoots to y~3e3, a worse
    # residual. With stall_patience=1 that non-improving step triggers a restart
    # from the best iterate (the start) at a tenth of dt, and the solve then
    # converges in fewer iterations than riding out the overshoot.
    def rhs(y, p):
        return 1.0 - y ** 3

    kw = dict(dt0=1e6, scale_floor=1.0, nonneg=False, divergence_factor=jnp.inf)
    held, *_ = ptc_forward(rhs, None, jnp.array([0.01]), max_iter=1, stall_patience=1, **kw)
    jumped, *_ = ptc_forward(rhs, None, jnp.array([0.01]), max_iter=1, stall_patience=0, **kw)
    assert float(held[0]) == pytest.approx(0.01)
    assert float(jumped[0]) > 1e3
    y, _, k_restart, conv = ptc_forward(rhs, None, jnp.array([0.01]), stall_patience=1,
                                        tol=1e-10, **kw)
    *_, k_ride, conv_ride = ptc_forward(rhs, None, jnp.array([0.01]), stall_patience=0,
                                        tol=1e-10, **kw)
    assert bool(conv) and bool(conv_ride)
    assert int(k_restart) < int(k_ride)
    np.testing.assert_allclose(float(y[0]), 1.0, atol=1e-8)


@pytest.mark.slow
def test_bsm2_steady_state_per_state_scaling_converges_competitively():
    # The default per-state pseudo-time / residual floor (max(|y0|, 1e-6)) gives
    # every state a magnitude-consistent scale, so the SER ramp is not throttled by
    # the over-damped small-magnitude states. With the old simple step-acceptance
    # guard this roughly halved the PTC iteration count vs the flat scalar floor;
    # the backtracking line-search PTC now does much of that step-size control
    # itself, so the per-state advantage is real but narrow (and the exact count is
    # platform-sensitive at the few-iteration level, since the line search's
    # accept/reject hinges on floating-point residual comparisons). So this guards
    # the two STABLE invariants -- both floors converge, to the SAME operating
    # point -- and that per-state scaling is not a material iteration regression,
    # rather than asserting a brittle strict iteration win.
    from aquakin.plant.bsm.bsm2 import (
        build_bsm2, bsm2_constant_influent, bsm2_parameters)
    from aquakin.plant.bsm import bsm2_warm_start
    asm1 = aquakin.load_model("asm1")
    adm1 = aquakin.load_model("adm1")
    # Pinned to the per-species settler without soluble holdup: this compares the
    # two scale floors, and needs a plant on which the flat floor converges. With
    # the default reference settler (lumped TSS + holdup) the flat floor still
    # stalls near the root on many starts (issue #627): the layers below the feed
    # all reach the same TSS, which puts the steady state on a kink of the Takacs
    # min(flux) between equal neighbours, and an unscaled Newton step chatters
    # across it. The per-state default converges there (see
    # test_bsm2_steady_state_matches_forward).
    plant = build_bsm2(asm1_model=asm1, adm1_model=adm1,
                       settler_composition_mode="per_species",
                       settler_soluble_holdup=False)
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    y0 = bsm2_warm_start(plant)
    params = bsm2_parameters(asm1, adm1)

    default = plant.steady_state(params, y0=y0)                # per-state floor
    flat = plant.steady_state(params, y0=y0, scale_floor=1.0)  # old behaviour
    assert bool(default.converged) and bool(flat.converged)
    # Competitive, not a regression: per-state scaling does not take materially
    # more iterations than the flat floor. The generous band tolerates the
    # few-iteration platform jitter while still failing loudly if a future change
    # makes per-state scaling pathological (e.g. doubling the count).
    assert int(default.iterations) <= int(flat.iterations) * 1.3 + 5
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
                     diff=aquakin.DifferentiationConfig(mode="reverse"))
    mine, theirs = dict(res.ranked("SNH")), dict(d.ranked())
    for s in screen:
        assert mine[s] == pytest.approx(theirs[s], rel=1e-6, abs=1e-12)

    # convergence (sample-size study): running bound + MC std error vs sample count
    counts, bound, se = res.convergence()
    assert bound.shape == (len(counts), 2, 4)
    assert se.shape == bound.shape
    j = res.input_names.index("asm1.muA")               # dominant input
    assert float(se[-1, 0, j]) < float(se[0, 0, j])     # std error shrinks with N


def test_steady_state_dgsm_normal_input():
    """The Gaussian-prior input path (``input_dist="normal"``) samples each
    parameter in its calibration-transform space (log-normal for a rate) and
    bounds the Sobol index with the sigma^2 Poincare constant in place of the
    uniform (b-a)^2/pi^2. It runs end-to-end, records the per-parameter Poincare
    constant on the result, and convergence() re-aggregates on it consistently."""
    plant, asm1, y0 = _bsm1()
    base = plant.default_parameters()
    si = asm1.species_index
    screen = ["asm1.muH", "asm1.muA", "asm1.bA"]
    idx = [plant.parameter_index(s) for s in screen]
    val = np.array([float(base[i]) for i in idx])
    ranges = np.array([[v * 0.7, v * 1.4] for v in val])     # read as a +-2sigma band
    transforms = [asm1.parameter_transforms.get(s.split(".")[1], "none")
                  for s in screen]

    def out_fn(y):
        return jnp.array([plant.states_by_unit(y)["tank5"][si["SNO"]]])

    res = plant.steady_state_dgsm(
        ranges, output_fn=out_fn, output_names=["SNO"], wrt=screen,
        n_samples=16, seed=0, y0=y0,
        input_dist="normal", input_transforms=transforms)
    assert res.poincare is not None and res.poincare.shape == (3,)
    # the band [lo,hi] is +-2sigma in transform space => sigma_z = (t(hi)-t(lo))/4,
    # so for a positive_log rate the Poincare constant is (log(1.4/0.7)/4)^2.
    expected = (np.log(1.4 / 0.7) / 4.0) ** 2
    assert np.allclose(np.asarray(res.poincare), expected, rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(res.sobol_total_bound)))

    # convergence() recomputes the bound from the retained per-sample data using
    # the SAME Poincare constant, so its last prefix matches the headline bound.
    counts, bound, se = res.convergence()
    assert np.allclose(np.asarray(bound[-1]),
                       np.asarray(res.sobol_total_bound), rtol=1e-6, equal_nan=True)

    # input_dist="normal" requires transforms aligned with the screened parameters
    with pytest.raises(ValueError):
        plant.steady_state_dgsm(ranges, output_fn=out_fn, wrt=screen,
                                n_samples=8, seed=0, y0=y0, input_dist="normal")


def test_dgsm_cond_filter_and_with_cond_factor():
    """The near-singular-Jacobian filter (the heavy-tail robustification) drops the
    samples it should and re-aggregation reuses the retained per-sample data. Fast
    and deterministic -- exercises the aggregation directly, no plant solve."""
    from aquakin.plant.sensitivity import (_dgsm_aggregate, _cond_mask,
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


@pytest.mark.slow
def test_bsm2_digester_forward_scaled_residual_matches_ptc():
    """On the BSM2 anaerobic digester (ADM1, 3400 m3, 35 C, the published feed),
    run_to_steady_state(criterion="scaled_residual") meets the PTC tolerance on
    the PTC criterion and lands on the PTC root. The default norm test, started
    from the published steady state, stops almost at once with a scaled residual
    many orders above it, which is what made the two look inconsistent."""
    from aquakin.plant import InfluentSeries, Plant
    from aquakin.plant.digester import ADM1DigesterUnit

    # The published BSM2 digester feed and steady state (Gernaey et al. 2014).
    q_feed, v_liq = 178.4674, 3400.0
    feed = {
        "S_aa": 0.04388, "S_IC": 0.0079326, "S_IN": 0.0019721, "S_I": 0.028067,
        "X_ch": 3.7236, "X_pr": 15.9235, "X_li": 8.047, "X_I": 17.0106,
        "S_an": 0.0052101,
    }
    ref_ss = {
        "S_su": 0.012394, "S_aa": 0.0055432, "S_fa": 0.10741, "S_va": 0.012333,
        "S_bu": 0.014003, "S_pro": 0.017584, "S_ac": 0.089315, "S_ch4": 0.05549,
        "S_IC": 0.095149, "S_IN": 0.094468, "S_I": 0.13087, "X_c": 0.10792,
        "X_ch": 0.020517, "X_pr": 0.08422, "X_li": 0.043629, "X_su": 0.31222,
        "X_aa": 0.93167, "X_fa": 0.33839, "X_c4": 0.33577, "X_pro": 0.10112,
        "X_ac": 0.67724, "X_h2": 0.28484, "X_I": 17.2162, "S_gas_ch4": 1.6535,
        "S_gas_co2": 0.01354,
    }
    adm1 = aquakin.load_model("adm1")
    si = adm1.species_index
    plant = Plant("digester")
    plant.add_unit(ADM1DigesterUnit(name="digester", model=adm1, volume=v_liq,
                                    conditions={"T": 308.15}, input_port_names=["feed"]))
    plant.add_influent("feed", InfluentSeries.constant(adm1, Q=q_feed, **feed),
                       to="digester.feed")
    C0 = np.asarray(adm1.default_concentrations()).copy()
    for k, v in ref_ss.items():
        C0[si[k]] = v
    y0 = plant.initial_state(overrides={"digester": jnp.asarray(C0)})

    tol, floor = 1e-8, 1e-6
    ptc = plant.steady_state(y0=y0, tol=tol, scale_floor=floor)
    assert ptc.method == "ptc" and ptc.converged and ptc.residual <= tol

    fwd = plant.run_to_steady_state(y0=y0, criterion="scaled_residual", tol=tol,
                                    scale_floor=floor)
    assert fwd.converged and fwd.residual <= tol
    a, b = np.asarray(ptc.state), np.asarray(fwd.state)
    big = np.abs(a) > 1e-3
    assert np.max(np.abs(b[big] - a[big]) / np.abs(a[big])) < 1e-6

    norm = plant.run_to_steady_state(y0=y0, scale_floor=floor)
    assert norm.converged and norm.residual > 1e3 * tol  # the gap the option closes
