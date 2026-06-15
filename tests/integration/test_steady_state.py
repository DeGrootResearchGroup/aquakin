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


@pytest.mark.slow
def test_bsm1_steady_state_matches_forward():
    plant, asm1, y0 = _bsm1()
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
def test_bsm1_steady_state_differentiable():
    # The steady state carries the IFT parameter gradient; check it against a
    # central finite difference in the heterotroph max-growth rate.
    plant, asm1, y0 = _bsm1()
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
def test_bsm1_steady_state_falls_back_to_forward():
    # If PTC is starved of iterations it falls back to the forward solve.
    plant, _asm1, y0 = _bsm1()
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
