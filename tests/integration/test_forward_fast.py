"""Lean forward-only integrator (``forward_solve`` / ``Plant.solve(forward_fast=
True)``).

A non-AD plant solve skips the diffrax adjoint / optimistix / lineax machinery
(whose purpose is differentiability) for a plain ``lax.while_loop`` ESDIRK. The
per-step Jacobian is still colored forward-mode AD -- the *same* exact matrix the
differentiable path uses -- so the trajectory matches a valid adaptive solution
to the same tolerance; only end-to-end differentiability is given up. These tests
pin the integrator math (analytic decay + order), the exact ``t_eval`` output,
the agreement with the diffrax solve on BSM1/BSM2, and the guards.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate.forward_solve import forward_solve


# --------------------------------------------------------------------------
# Synthetic (fast, no plant) -- the integrator math.
# --------------------------------------------------------------------------

def test_decay_matches_analytic_and_saveat():
    """y' = -k y has y(t) = y0 exp(-k t). Check the exact t_eval output and the
    initial point recorded exactly."""
    k = jnp.array([2.0, 0.5, 5.0])
    y0 = jnp.array([1.0, 2.0, 0.5])

    def rhs(t, y, args):
        return -k * y

    def jac(t, y, args):
        return jnp.diag(-k)

    t_eval = jnp.linspace(0.0, 1.5, 7)
    ys = forward_solve(rhs, jac, y0, None, 0.0, 1.5, t_eval,
                       rtol=1e-7, atol=1e-10)
    exact = y0[None, :] * jnp.exp(-k[None, :] * t_eval[:, None])
    assert np.allclose(np.asarray(ys[0]), np.asarray(y0))      # t0 exact
    assert np.max(np.abs(np.asarray(ys) - np.asarray(exact))) < 1e-6
    assert not np.any(np.all(np.asarray(ys) == 0, axis=1))     # no missed points


def test_dense_output_save_grid_independent():
    """Dense output (cubic-Hermite continuous extension, #386): the integrator
    takes its natural steps (clipped only to t1, not to each save) and interpolates
    save points within a step. So the solution at a save time does NOT depend on
    how dense the t_eval grid is -- a sparse and a dense grid are interpolated from
    the *same* step sequence and must agree exactly at shared times. (With the old
    step-clipping every save forced a step boundary, so the grids gave different
    step sequences and only agreed to integration tolerance.)"""
    k = jnp.array([1.0, 0.3, 2.5])
    y0 = jnp.array([1.0, 2.0, 0.5])

    def rhs(t, y, args):
        return -k * y

    def jac(t, y, args):
        return jnp.diag(-k)

    dense = jnp.linspace(0.0, 4.0, 201)         # 0.02 spacing
    sparse = jnp.linspace(0.0, 4.0, 5)          # 0,1,2,3,4 -- a subset of dense
    yd = forward_solve(rhs, jac, y0, None, 0.0, 4.0, dense, rtol=1e-4, atol=1e-8)
    ys = forward_solve(rhs, jac, y0, None, 0.0, 4.0, sparse, rtol=1e-4, atol=1e-8)
    shared = np.asarray(yd)[::50]               # dense rows at t = 0,1,2,3,4
    # Same step sequence interpolated at the same times -> bit-for-bit agreement.
    assert np.max(np.abs(shared - np.asarray(ys))) < 1e-11
    # And the interpolation tracks the analytic decay (most dense points fall
    # strictly inside a step at this loose tolerance).
    exact = np.asarray(y0)[None, :] * np.exp(-np.asarray(k)[None, :]
                                             * np.asarray(dense)[:, None])
    assert np.max(np.abs(np.asarray(yd) - exact)) < 1e-3
    assert not np.any(np.all(np.asarray(yd) == 0, axis=1))     # no missed points


def test_order_is_three():
    """Kvaerno3 is 3rd order: error should drop ~8x per halving of tolerance-
    driven step (checked via a stiff scalar with a tight vs loose tol)."""
    def rhs(t, y, args):
        return -10.0 * (y - jnp.sin(t))

    def jac(t, y, args):
        return jnp.array([[-10.0]])

    y0 = jnp.array([0.0])
    te = jnp.array([0.0, 2.0])
    truth = forward_solve(rhs, jac, y0, None, 0.0, 2.0, te,
                          rtol=1e-11, atol=1e-12)[-1]
    e_loose = abs(float(forward_solve(rhs, jac, y0, None, 0.0, 2.0, te,
                                      rtol=1e-4, atol=1e-7)[-1, 0] - truth[0]))
    e_tight = abs(float(forward_solve(rhs, jac, y0, None, 0.0, 2.0, te,
                                      rtol=1e-6, atol=1e-9)[-1, 0] - truth[0]))
    assert e_tight < e_loose                                   # converges with tol


# --------------------------------------------------------------------------
# Plant guards (fast: reject before any solve).
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bsm1():
    from aquakin.plant.bsm import build_bsm1, bsm1_warm_start
    from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
    from aquakin.plant.influent import InfluentSeries

    def make():
        asm1 = aquakin.load_network("asm1")
        p = build_bsm1(network=asm1)
        C0 = asm1.concentrations({
            "SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
            "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0})
        inf = InfluentSeries(t=jnp.array([0.0, 100.0]),
                             Q=jnp.full((2,), BSM1_Q_AVG),
                             C=jnp.tile(C0, (2, 1)), network=asm1)
        p.add_influent("feed", inf, to="inlet_mix.fresh")
        return p, bsm1_warm_start(p)

    return make


def test_forward_fast_rejects_events_grad_stable_adjoint(bsm1):
    plant, y0 = bsm1()
    params = plant.default_parameters()
    te = jnp.array([1.0])
    with pytest.raises(ValueError, match="forward_fast"):
        plant.solve(t_span=(0.0, 1.0), t_eval=te, params=params, y0=y0,
                    forward_fast=True, events=[aquakin.Event(at_times=[0.5])])
    with pytest.raises(ValueError, match="forward_fast"):
        plant.solve(t_span=(0.0, 1.0), t_eval=te, params=params, y0=y0,
                    forward_fast=True, gradient="stable_adjoint")
    # not differentiable: a reverse-mode trace must raise the concrete-input error
    with pytest.raises(ValueError, match="forward_fast requires concrete"):
        jax.grad(lambda s: jnp.sum(plant.solve(
            t_span=(0.0, 1.0), t_eval=te, params=params * s, y0=y0,
            forward_fast=True).state[-1]))(1.0)


# --------------------------------------------------------------------------
# Plant agreement (slow: full solves).
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_forward_fast_matches_diffrax_bsm1(bsm1):
    plant, y0 = bsm1()
    params = plant.default_parameters()
    te = jnp.linspace(0.0, 8.0, 17)
    kw = dict(t_span=(0.0, 8.0), t_eval=te, params=params, y0=y0,
              rtol=1e-5, atol=1e-3, max_steps=2_000_000)
    ff = plant.solve(**kw, forward_fast=True)
    ref = plant.solve(**kw)
    a = np.asarray(ff.state)
    b = np.asarray(ref.state)
    assert ff.state.shape == ref.state.shape
    assert np.all(np.isfinite(a))
    assert np.allclose(a[0], b[0])                              # t0 exact
    rel = np.max(np.abs(a - b) / (np.abs(b) + 1e-3))
    assert rel < 1e-2          # two valid adaptive solutions to the same rtol


@pytest.mark.slow
def test_forward_fast_matches_diffrax_bsm2():
    from aquakin.plant.bsm import bsm2_warm_start
    from aquakin.plant.bsm.bsm2 import (
        InfluentBypass, build_bsm2, bsm2_asm1_network, bsm2_parameters)
    from aquakin.plant.influent import load_bsm2_influent
    asm1 = bsm2_asm1_network(); adm1 = aquakin.load_network("adm1")
    params = bsm2_parameters(asm1, adm1)
    p = build_bsm2(asm1_network=asm1, adm1_network=adm1,
                   do_temperature_correction=True,
                   bypass=InfluentBypass(threshold=60000.0))
    p.add_influent("feed", load_bsm2_influent("dry", asm1))
    y0 = bsm2_warm_start(p)
    te = jnp.linspace(0.0, 8.0, 17)
    kw = dict(t_span=(0.0, 8.0), t_eval=te, params=params, y0=y0,
              rtol=1e-4, atol=1e-3, max_steps=8_000_000)
    ff = p.solve(**kw, forward_fast=True)
    ref = p.solve(**kw, colored_jacobian=True)
    a, b = np.asarray(ff.state), np.asarray(ref.state)
    assert p._colored_root_finder[2] is True                   # colored guard ok
    assert np.all(np.isfinite(a))
    assert np.allclose(a[0], b[0])
    rel = np.max(np.abs(a - b) / (np.abs(b) + 1e-3))
    assert rel < 1.5e-2
