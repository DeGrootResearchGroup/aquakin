"""Adaptive AD-safe recycle-concentration resolution (``recycle_tol``).

The recycle back-edge streams are the fixed point ``x = G(x)`` of one forward
output sweep (flow ``Q``, concentration ``C``, temperature ``T`` read back on the
edges). The default fixed ``recycle_passes`` Gauss-Seidel mop-up converges in
``log(tol)/log(rho)`` passes, where ``rho`` is the nonlinear flow<->concentration
coupling's spectral radius -- a *topology-dependent* number, low for the BSM
reject loop (~0.0066) but not bounded below 1 for an arbitrary recycle-heavy
plant. ``recycle_tol`` replaces the fixed count with an adaptive
``lax.while_loop`` (warm-started from the exact affine seed) wrapped in
``jax.lax.custom_root``: it iterates until the *actual* residual clears, so it is
correct for any ``rho < 1``, and the gradient is the exact implicit-function-
theorem tangent.

The synthetic tests drive :meth:`Plant._recycle._adaptive_recycle_refine` directly with a
controllable-``rho`` map (fast, no plant solve) to pin the generality guarantee
and the gradient; the BSM2 tests pin it on a real two-network plant.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant.plant import Plant


# ---------------------------------------------------------------------------
# Fast (PR-gate) synthetic tests: drive _adaptive_recycle_refine directly with a
# tunable-rho fixed-point map. No plant solve, so these run on the fast gate.
# ---------------------------------------------------------------------------

_KEY = ("u", "p")
_KEYS = [_KEY]


def _synthetic_forward(theta, rho):
    """One-pass recycle map with spectral radius ``rho`` (a contraction for
    ``rho < 1``). C follows ``rho*c + theta + 0.05*tanh(c)`` (nonlinear, so the
    affine seed is off the fixed point); Q is reject-like, coupled to C (a
    non-degenerate self-map, as a real concentration-dependent underflow is)."""
    def forward_full(q, c, T):
        cn = rho * c[_KEY] + theta + 0.05 * jnp.tanh(c[_KEY])
        qn = 10.0 + 0.1 * jnp.sum(c[_KEY])
        return {_KEY: cn}, {_KEY: None}, {_KEY: qn}
    return forward_full


def _true_C_fixed_point(theta, rho, n=4000):
    x = jnp.zeros_like(theta)
    for _ in range(n):
        x = rho * x + theta + 0.05 * jnp.tanh(x)
    return x


@pytest.mark.parametrize("rho", [0.5, 0.9, 0.99])
def test_adaptive_converges_for_high_rho(rho):
    """The adaptive solve reaches tolerance for any rho < 1, where a fixed
    3-pass mop-up leaves a residual of order rho**3 (catastrophic as rho->1).

    This is the generality guarantee: the "3 passes is enough" calibration is
    specific to the low-gain BSM reject loop; a high-gain recycle needs the
    adaptive solve, which the fixed count cannot provide.
    """
    theta = jnp.array([1.0, 2.0, 3.0])
    fwd = _synthetic_forward(theta, rho)
    p = Plant("t")
    p.recycle_tol = 1e-12
    p.recycle_max_passes = 5000
    seed_C = {_KEY: jnp.zeros(3)}  # deliberately far from the fixed point
    _, Cr, _ = p._recycle._adaptive_recycle_refine(
        fwd, _KEYS, {_KEY: jnp.array(5.0)}, seed_C, {_KEY: None}, False)
    xstar = _true_C_fixed_point(theta, rho)
    err_adaptive = float(np.max(np.abs(Cr[_KEY] - xstar)) / np.max(np.abs(xstar)))
    assert err_adaptive < 1e-7

    # A fixed 3-pass mop-up from the same seed is far from converged.
    x3 = jnp.zeros(3)
    for _ in range(3):
        x3 = rho * x3 + theta + 0.05 * jnp.tanh(x3)
    err_fixed3 = float(np.max(np.abs(x3 - xstar)) / np.max(np.abs(xstar)))
    # The adaptive solve is orders of magnitude tighter; the gap grows with rho.
    assert err_fixed3 > 100 * err_adaptive


def test_adaptive_ift_gradient_matches_finite_difference():
    """The custom_root IFT tangent of the adaptive fixed point matches central
    finite differences -- exact, and O(1) in the iteration count."""
    rho = 0.9
    seed = {_KEY: jnp.zeros(3)}

    def loss(theta):
        p = Plant("t")
        p.recycle_tol = 1e-13
        p.recycle_max_passes = 5000
        fwd = _synthetic_forward(theta, rho)
        _, Cr, _ = p._recycle._adaptive_recycle_refine(
            fwd, _KEYS, {_KEY: jnp.array(5.0)}, dict(seed), {_KEY: None}, False)
        return jnp.sum(Cr[_KEY] ** 2)

    theta = jnp.array([1.0, 2.0, 3.0])
    g = np.asarray(jax.grad(loss)(theta))
    assert np.all(np.isfinite(g))
    eps = 1e-6
    gfd = np.array([
        (float(loss(theta.at[i].add(eps))) - float(loss(theta.at[i].add(-eps))))
        / (2 * eps) for i in range(3)])
    rel = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-30)
    assert rel < 1e-6


def test_recycle_tol_construction_validation():
    """``recycle_tol`` / ``recycle_max_passes`` are validated at construction."""
    assert Plant("a").recycle_tol == 1e-8            # adaptive on by default
    assert Plant("b", recycle_tol=1e-9).recycle_tol == 1e-9
    assert Plant("f", recycle_tol=None).recycle_tol is None  # opt out -> fixed-pass
    with pytest.raises(ValueError):
        Plant("c", recycle_tol=0.0)
    with pytest.raises(ValueError):
        Plant("d", recycle_tol=-1e-3)
    with pytest.raises(ValueError):
        Plant("e", recycle_max_passes=0)


# ---------------------------------------------------------------------------
# Slow (merge-tier) BSM2 tests: the adaptive solve on a real two-network plant
# reaches the same fixed point as a deep fixed-pass sweep, forward and gradient.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bsm2():
    from aquakin.plant.bsm import bsm2_warm_start, bsm2_constant_influent
    from aquakin.plant.bsm.bsm2 import (
        build_bsm2, bsm2_asm1_network, bsm2_parameters,
        BSM2_CONSTANT_INFLUENT_T)
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)
    p = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    p.add_influent("feed", bsm2_constant_influent(asm1, T=BSM2_CONSTANT_INFLUENT_T))
    y0 = bsm2_warm_start(p)
    # Prime the layouts / caches with a tiny solve.
    p.solve(t_span=(0.0, 0.02), t_eval=jnp.array([0.02]), params=params, y0=y0,
            rtol=1e-3, atol=1e-2,
            integrator=aquakin.IntegratorConfig(max_steps=100_000))
    return p, params, y0


@pytest.mark.slow
def test_bsm2_adaptive_matches_deep_sweep(bsm2):
    """Adaptive (tol=1e-12) recycle resolution == a 14-pass fixed sweep, forward
    streams (the true nonlinear reject-loop fixed point)."""
    p, params, y0 = bsm2
    t0 = jnp.asarray(0.0)
    pf = p._coerce_params(params)
    # A perturbed state exercises the concentration-dependent reject flows.
    yp = jnp.asarray(np.asarray(y0) * 1.3 + 1.0)
    states = p._split_state(yp)
    sig = p._compute_signals(t0, states, pf)
    flows = p._recycle._resolve_flows(t0, pf, states)
    influent = {(None, pn): s.at(t0) for pn, s in p.influents.items()}

    p.recycle_tol = None
    seed_aff = p._recycle._resolve_recycle_concentrations(t0, states, pf, flows, sig)
    deep = p._sweep_outputs(t0, states, dict(influent), seed_aff, pf,
                            passes=14, signals=sig)

    p.recycle_tol = 1e-12
    seed_ad = p._recycle._resolve_recycle_concentrations(t0, states, pf, flows, sig)
    out_ad = p._sweep_outputs(t0, states, dict(influent), seed_ad, pf,
                              passes=1, signals=sig)
    p.recycle_tol = None

    worst = 0.0
    for k in out_ad:
        ca = np.asarray(out_ad[k].C)
        cd = np.asarray(deep[k].C)
        if ca.size == 0:
            continue
        worst = max(worst, np.max(np.abs(ca - cd)) / (np.max(np.abs(cd)) + 1e-12))
    assert worst < 1e-9


@pytest.mark.slow
def test_bsm2_adaptive_ift_tangent_matches_deep(bsm2):
    """The adaptive recycle gradient (custom_root IFT) w.r.t. the state matches a
    deep fixed-pass gradient, and is finite and non-trivial."""
    p, params, y0 = bsm2
    t0 = jnp.asarray(0.0)
    pf = p._coerce_params(params)
    yp = jnp.asarray(np.asarray(y0) * 1.3 + 1.0)
    influent = {(None, pn): s.at(t0) for pn, s in p.influents.items()}

    def loss(y):
        st = p._split_state(y)
        s = p._compute_signals(t0, st, pf)
        fl = p._recycle._resolve_flows(t0, pf, st)
        seed = p._recycle._resolve_recycle_concentrations(t0, st, pf, fl, s)
        outs = p._sweep_outputs(t0, st, dict(influent), seed, pf, signals=s)
        return jnp.sum(outs[("front_mix", "out")].C)

    p.recycle_tol = 1e-12
    g_ad = jax.grad(loss)(yp)
    p.recycle_tol = None
    p.recycle_passes = 14
    g_deep = jax.grad(loss)(yp)
    p.recycle_passes = 3

    assert bool(np.all(np.isfinite(g_ad)))
    assert float(np.max(np.abs(g_ad))) > 1.0   # non-trivial
    rel = float(np.max(np.abs(g_ad - g_deep)) / (np.max(np.abs(g_deep)) + 1e-30))
    assert rel < 1e-6


@pytest.mark.slow
def test_bsm2_adaptive_forward_solve_matches_fixed(bsm2):
    """A short BSM2 solve with recycle_tol set matches the fixed-pass default
    trajectory (both converge to the recycle fixed point)."""
    p, params, y0 = bsm2
    te = jnp.linspace(0.0, 3.0, 13)
    p.recycle_tol = None
    p._jit_cache.clear()
    s_fx = np.asarray(p.solve(
        t_span=(0.0, 3.0), t_eval=te, params=params, y0=y0,
        rtol=1e-6, atol=1e-5,
        integrator=aquakin.IntegratorConfig(max_steps=400_000)).state[-1])
    p.recycle_tol = 1e-10
    p._jit_cache.clear()
    s_ad = np.asarray(p.solve(
        t_span=(0.0, 3.0), t_eval=te, params=params, y0=y0,
        rtol=1e-6, atol=1e-5,
        integrator=aquakin.IntegratorConfig(max_steps=400_000)).state[-1])
    p.recycle_tol = None
    p._jit_cache.clear()
    rel = float(np.max(np.abs(s_ad - s_fx)) / (np.max(np.abs(s_fx)) + 1e-9))
    assert rel < 1e-5
