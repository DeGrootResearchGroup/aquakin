"""Lean forward-only integrator (``forward_solve`` / ``Plant.solve(forward_fast=
True)``).

A non-AD plant solve skips the diffrax adjoint / optimistix / lineax machinery
(whose purpose is differentiability) for a plain ``lax.while_loop`` ESDIRK. The
per-step Jacobian is still colored forward-mode AD -- the *same* exact matrix the
differentiable path uses -- so the trajectory matches a valid adaptive solution
to the same tolerance; only end-to-end differentiability is given up. These tests
pin the integrator math (analytic decay + order), the exact ``t_eval`` output,
the agreement with the diffrax solve on BSM2, and the guards (on the
one-tank plant).
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.integrate.forward_solve import StepControl, forward_solve

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
    ys, _ = forward_solve(rhs, jac, y0, None, 0.0, 1.5, t_eval,
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
    yd, _ = forward_solve(rhs, jac, y0, None, 0.0, 4.0, dense, rtol=1e-4, atol=1e-8)
    ys, _ = forward_solve(rhs, jac, y0, None, 0.0, 4.0, sparse, rtol=1e-4, atol=1e-8)
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
    """Kvaerno3 is third order: at a fixed step ``h`` the global error falls as
    ``h**3``, so halving ``h`` divides it by 8.

    ``y' = -10 (y - sin t)``, ``y(0) = 0`` has the closed form
    ``y = (10 sin t - cos t + exp(-10 t)) * 10/101``. The step is pinned by
    ``h0 = dtmax = h`` with tolerances too loose to reject a step; the problem
    is linear and the Jacobian exact, so each stage solve is exact and only the
    method's truncation error remains. The observed order
    ``log2(e(h) / e(h/2))`` approaches 3 from below as ``h`` shrinks (2.89,
    2.94, 2.97 for ``h`` = 0.02, 0.01, 0.005): the next term of the error
    expansion is one power of ``h`` smaller. At ``h = 0.01`` it is within 0.1
    of 3; a second- or fourth-order method would be about 1 away.
    """
    def rhs(t, y, args):
        return -10.0 * (y - jnp.sin(t))

    def jac(t, y, args):
        return jnp.array([[-10.0]])

    t1 = 2.0
    exact = 10.0 / 101.0 * (10.0 * np.sin(t1) - np.cos(t1) + np.exp(-10.0 * t1))

    def error(h):
        ys, _ = forward_solve(rhs, jac, jnp.array([0.0]), None, 0.0, t1,
                           jnp.array([0.0, t1]), rtol=1e3, atol=1e3,
                           control=StepControl(h0=h, dtmax=h))
        return abs(float(ys[-1, 0]) - exact)

    order = np.log2(error(0.01) / error(0.005))
    assert abs(order - 3.0) < 0.1


def test_step_budget_exhaustion_raises():
    """Running out of steps must report it, not return zero-filled rows: the
    result is ``max_steps_reached``, the unreached rows are NaN, and
    ``check_solve_result`` raises the step-budget error."""
    import diffrax

    from aquakin.integrate._solver import check_solve_result

    def rhs(t, y, args):
        return -args[0] * y

    def jac(t, y, args):
        return -args[0] * jnp.eye(1)

    te = jnp.linspace(0.0, 10.0, 5)
    ys, result = forward_solve(rhs, jac, jnp.array([1.0]), jnp.array([0.3]), 0.0, 10.0, te,
                               rtol=1e-6, atol=1e-9, control=StepControl(max_steps=3))
    assert bool(result == diffrax.RESULTS.max_steps_reached)
    assert np.isnan(np.asarray(ys[-1])).all()
    with pytest.raises(RuntimeError, match=r"step budget \(max_steps=3\)"):
        check_solve_result(result, ys, max_steps=3, what="test solve")


def test_stalled_solve_raises():
    """A solution that blows up (y' = y^2 from y0 = 1, singular at t = 1) stalls
    the step size; the solve must report it rather than return zeros."""
    import diffrax

    from aquakin.integrate._solver import check_solve_result

    def rhs(t, y, args):
        return y**2

    def jac(t, y, args):
        return 2.0 * y * jnp.eye(1)

    te = jnp.linspace(0.0, 2.0, 5)
    ys, result = forward_solve(rhs, jac, jnp.array([1.0]), None, 0.0, 2.0, te,
                               rtol=1e-6, atol=1e-9,
                               control=StepControl(max_steps=100_000))
    assert bool(result == diffrax.RESULTS.dt_min_reached)
    with pytest.raises(eqx.EquinoxRuntimeError, match="minimum step size"):
        check_solve_result(result, ys, max_steps=100_000, what="test solve")


# --------------------------------------------------------------------------
# Plant guards and knobs, on the one-tank plant: they do not depend on the
# plant's size.
# --------------------------------------------------------------------------

_Y0 = jnp.array([0.2, 0.0])


def test_forward_fast_rejects_events_and_grad(cstr_plant):
    plant, y0 = cstr_plant, _Y0
    params = plant.default_parameters()
    te = jnp.array([1.0])
    with pytest.raises(ValueError, match="forward_fast"):
        plant.solve(t_span=(0.0, 1.0), t_eval=te, params=params, y0=y0,
                    forward_fast=True, events=[aquakin.Event(at_times=[0.5])])
    # forward_fast is a plain forward solve, so it composes with the DEFAULT diff
    # (method="stable"); the incompatibility is with DIFFERENTIATING it, which the
    # concrete-input guard below enforces (not the diff config on a plain solve).
    plant.solve(t_span=(0.0, 1.0), t_eval=te, params=params, y0=y0,
                forward_fast=True,
                diff=aquakin.DifferentiationConfig(method="stable"))
    # not differentiable: a reverse-mode trace must raise the concrete-input error
    with pytest.raises(ValueError, match="forward_fast requires concrete"):
        jax.grad(lambda s: jnp.sum(plant.solve(
            t_span=(0.0, 1.0), t_eval=te, params=params * s, y0=y0,
            forward_fast=True).state[-1]))(1.0)


# The integrator knobs forward_fast cannot thread through its hand-rolled ESDIRK
# -- each paired with a non-default value. forward_fast must REJECT every one of
# these, never silently drop it (the silent-footgun class, issue #446). dtmax is
# paired with method="through_solve" because the default stable method rejects an
# explicit dtmax even earlier (also correct -- the point is no silent drop).
@pytest.mark.parametrize("override, diff", [
    ({"solver": diffrax.Kvaerno5()}, None),
    ({"order": 5}, None),
    ({"factormax": 1.0}, None),
    ({"factormax": None}, None),
    ({"colored_jacobian": True}, None),
    ({"colored_jacobian": False}, None),
    ({"dtmax": 0.1}, aquakin.DifferentiationConfig(method="through_solve")),
])
def test_forward_fast_rejects_unhonored_integrator_knobs(cstr_plant, override, diff):
    plant, y0 = cstr_plant, _Y0
    params = plant.default_parameters()
    te = jnp.array([1.0])
    kw = {} if diff is None else {"diff": diff}
    with pytest.raises(ValueError, match="forward_fast cannot honour integrator"):
        plant.solve(t_span=(0.0, 1.0), t_eval=te, params=params, y0=y0,
                    forward_fast=True,
                    integrator=aquakin.IntegratorConfig(**override), **kw)


# The honored knobs -- rtol / atol / max_steps -- must NOT be rejected: a
# forward_fast solve that tunes only those validates and runs. (max_steps is the
# one integrator field forward_fast threads through; the other knob a forward_fast
# user routinely sets.) This is the "honors, not silently drops" half of the
# (mode x knob) contract.
def test_forward_fast_accepts_honored_knobs(cstr_plant):
    """A forward_fast solve that sets rtol / atol / max_steps runs, and matches
    the default diffrax solve at the same tolerances and the closed form.

    The tank's ``A`` obeys ``dA/dt = 0.1 (1 - A) - k A`` with ``k = 0.1``, so
    ``A(t) = 0.5 + (A0 - 0.5) exp(-0.2 t)``. Two adaptive solutions at
    ``rtol = 1e-7`` agree with each other and with the closed form to well
    within ``1e-5``, two decades above the tolerance.
    """
    plant, y0 = cstr_plant, _Y0
    params = plant.default_parameters()
    te = jnp.linspace(0.0, 10.0, 6)
    kw = dict(t_span=(0.0, 10.0), t_eval=te, params=params, y0=y0,
              rtol=1e-7, atol=1e-10,
              integrator=aquakin.IntegratorConfig(max_steps=2_000_000))
    fast = np.asarray(plant.solve(**kw, forward_fast=True).state)
    reference = np.asarray(plant.solve(**kw).state)
    np.testing.assert_array_equal(fast[0], np.asarray(y0))         # t0 exact
    np.testing.assert_allclose(fast, reference, rtol=1e-5, atol=1e-8)
    exact_A = 0.5 + (float(y0[0]) - 0.5) * np.exp(-0.2 * np.asarray(te))
    np.testing.assert_allclose(fast[:, 0], exact_A, rtol=1e-5)


def test_forward_fast_honours_max_steps(cstr_plant):
    """The plant's ``max_steps`` reaches the forward_fast integrator."""
    plant, y0 = cstr_plant, _Y0
    params = plant.default_parameters()
    with pytest.raises(RuntimeError, match="step budget"):
        plant.solve(t_span=(0.0, 1.0), t_eval=jnp.array([1.0]), params=params,
                    y0=y0, forward_fast=True,
                    integrator=aquakin.IntegratorConfig(max_steps=3))


# --------------------------------------------------------------------------
# Plant agreement (slow: full solves).
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_forward_fast_matches_diffrax_bsm2():
    from aquakin.plant.bsm import bsm2_warm_start
    from aquakin.plant.bsm.bsm2 import InfluentBypass, bsm2_asm1_model, bsm2_parameters, build_bsm2
    from aquakin.plant.influent import load_bsm2_influent
    asm1 = bsm2_asm1_model()
    adm1 = aquakin.load_model("adm1")
    params = bsm2_parameters(asm1, adm1)
    p = build_bsm2(asm1_model=asm1, adm1_model=adm1,
                   do_temperature_correction=True,
                   bypass=InfluentBypass(threshold=60000.0))
    p.add_influent("feed", load_bsm2_influent("dry", asm1))
    y0 = bsm2_warm_start(p)
    te = jnp.linspace(0.0, 8.0, 17)
    kw = dict(t_span=(0.0, 8.0), t_eval=te, params=params, y0=y0,
              rtol=1e-4, atol=1e-3)
    ff = p.solve(**kw, forward_fast=True,
                 integrator=aquakin.IntegratorConfig(max_steps=8_000_000))
    ref = p.solve(**kw, integrator=aquakin.IntegratorConfig(
        max_steps=8_000_000, colored_jacobian=True))
    a, b = np.asarray(ff.state), np.asarray(ref.state)
    assert p._colored._root_finder[2] is True                   # colored guard ok
    assert np.all(np.isfinite(a))
    assert np.allclose(a[0], b[0])
    rel = np.max(np.abs(a - b) / (np.abs(b) + 1e-3))
    assert rel < 1.5e-2
