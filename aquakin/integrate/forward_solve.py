"""Lean forward-only adaptive ESDIRK integrator (no autodiff through the solve).

A stiff implicit solve through ``diffrax`` carries machinery whose purpose is to
make the *whole solve* differentiable -- an ``optimistix`` root finder, a
``lineax`` linear-solve abstraction, and a checkpointing reverse-mode adjoint
(``custom_vjp``). Tracing all of that dominates compile time (the implicit
scaffolding traces ~10x slower than the bare ODE loop), and it adds runtime
overhead. A **forward-only** plant solve -- one that never needs
``jax.grad`` / ``calibrate`` / ``sensitivity`` of the result -- can skip all of
it: a plain ``lax.while_loop`` running the Kvaerno3 stages with a simplified
Newton iteration and a direct dense ``lu_factor`` / ``lu_solve``.

The per-step Jacobian ``J = df/dy`` is **still computed by autodiff** (forward-mode
``jax.linearize``, materialised by the colored-AD column compression -- the same
exact Jacobian the differentiable path uses, so the step behaviour matches). What
is dropped is only the *adjoint over the whole solve*: the result is **not**
differentiable w.r.t. parameters / initial conditions. So this is an opt-in fast
lane for non-AD solves; any differentiated solve must use the ``diffrax`` path.

The compile saving is real but **scale-dependent**: it is large for small/simple
networks (where the ``diffrax`` implicit scaffolding dominates the trace), but on
a large flowsheet the dominant compile cost is the shared colored-Jacobian +
plant-RHS tracing, so the saving shrinks (measured only ~1.1x on the full BSM2
plant). The run is **not** faster than the colored ``diffrax`` path on a large
plant -- it is ~1.2x slower there, because ``diffrax``'s mature chord/PID solver
places steps more efficiently (the per-step costs are otherwise the same: both
build the colored Jacobian once per step, which XLA fuses into one batched JVP, so
freezing/reusing it does not help and in fact hurts via degraded Newton
convergence). So this is a fast lane for **small/simple** non-AD forward solves
and compile-bound repeated solves; for a large differentiable plant the
``colored_jacobian`` ``diffrax`` path is both faster and differentiable. Results
match the ``diffrax`` trajectory to the same ``rtol`` (the usual step-sequence
variation between two valid adaptive solves).

The method is the L-stable A-stable 3rd-order Kvaerno3 ESDIRK with its embedded
2nd-order error estimate (Kvaerno 2004) and an adaptive step controller with two
parts: an error controller -- the I-term ``en^(-1/3)`` with a Soderlind PI memory
term ``(en_prev/en)^pi_beta`` (the proportional part damps the overshoot-then-
reject oscillation a bare I-controller suffers; ``pi_beta=0`` recovers the pure
I-controller) -- and a **Gustafsson-style convergence-aware growth limiter** (the
simplified-Newton contraction rate caps the step growth, so a step is rarely grown
into nonlinear-divergence, the chord's failure mode; diffrax's ``PIDController``
has no such term). The default ``maxfac=2.0`` / ``pi_beta=0.08`` were tuned on the
full 609-day BSM2 dynamic run, where they cut the step count ~20% versus the
earlier ``maxfac=5`` pure-I controller (fewer error-test rejections). Output at
``t_eval`` is exact: the step is clipped to land on each save time (no
dense-output interpolation error).
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla


# --- Kvaerno3 tableau (Kvaerno 2004; coefficients as in diffrax) -------------
_G = 0.43586652150
_A21 = _G
_A31 = (-4 * _G**2 + 6 * _G - 1) / (4 * _G)
_A32 = (-2 * _G + 1) / (4 * _G)
_A41 = (6 * _G - 1) / (12 * _G)
_A42 = -1 / ((24 * _G - 12) * _G)
_A43 = (-6 * _G**2 + 6 * _G - 1) / (6 * _G - 3)
_TH = 1 / (2 * _G)
_AL31, _AL32 = 1.0 - _TH, _TH            # stage-3 predictor
_AL41, _AL42, _AL43 = _A31, _A32, _G     # stage-4 predictor
_C2, _C3, _C4 = 2 * _G, 1.0, 1.0         # stage times
_B = (_A41, _A42, _A43, _G)              # solution weights (stiffly accurate)
_BE = (_A41 - _A31, _A42 - _A32, _A43 - _G, _G)   # embedded-error weights

_MAXNEWT = 12
_KAPPA = 1e-2          # simplified-Newton convergence tolerance (Hairer eta test)


def forward_solve(
    rhs: Callable,
    jac: Callable,
    y0: jnp.ndarray,
    args,
    t0: float,
    t1: float,
    t_eval: jnp.ndarray,
    *,
    rtol: float,
    atol,
    h0: float = 1e-3,
    theta_target: float = 0.3,
    safety: float = 0.9,
    minfac: float = 0.2,
    maxfac: float = 2.0,
    pi_beta: float = 0.08,
    max_steps: int = 100_000_000,
):
    """Integrate ``dy/dt = rhs(t, y, args)`` from ``t0`` to ``t1``, forward only.

    Parameters
    ----------
    rhs : Callable
        ``rhs(t, y, args) -> dy/dt``.
    jac : Callable
        ``jac(t, y, args) -> J`` the dense ``(n, n)`` Jacobian ``df/dy`` (formed
        by colored forward AD by the caller).
    y0 : jnp.ndarray
        Initial state ``(n,)``.
    args :
        Threaded to ``rhs`` / ``jac`` (the parameter vector) -- a runtime
        argument, so one compile serves a parameter sweep.
    t0, t1 : float
        Integration interval.
    t_eval : jnp.ndarray
        Strictly increasing save times in ``[t0, t1]``; the solution is returned
        at exactly these times (steps are clipped to land on them).
    rtol, atol : float or array
        Error tolerances (WRMS norm ``atol + rtol*|y|``).

    Returns
    -------
    ys : jnp.ndarray
        Solution at ``t_eval``, shape ``(len(t_eval), n)``.
    """
    n = y0.shape[0]
    eye = jnp.eye(n)
    atol = jnp.asarray(atol)
    t_eval = jnp.asarray(t_eval)
    n_save = t_eval.shape[0]

    def wrms(v, y):
        scale = atol + rtol * jnp.abs(y)
        return jnp.sqrt(jnp.mean((v / scale) ** 2))

    def stage(t_s, base, k_init, lu, h, y):
        """Simplified Newton for ``k = rhs(t_s, base + h*G*k, args)`` with frozen
        operator ``lu = LU(I - h*G*J)``. Hairer-Wanner (ODEs II, IV.8) contraction
        test: converge when ``eta*||dk|| < KAPPA``; diverge when rate >= 1."""
        def cond(c):
            k, dprev, it, rate, conv, div = c
            return (~conv) & (~div) & (it < _MAXNEWT)

        def body(c):
            k, dprev, it, rate, conv, div = c
            g = k - rhs(t_s, base + h * _G * k, args)
            dk = jsla.lu_solve(lu, g)
            k = k - dk
            d = wrms(dk, y)
            rate = jnp.where(it >= 1, d / (dprev + 1e-30), rate)
            eta = rate / jnp.maximum(1.0 - rate, 1e-3)
            conv = (it >= 1) & (rate < 1.0) & (eta * d < _KAPPA)
            div = ((it >= 1) & (rate >= 1.0)) | (~jnp.isfinite(d))
            return k, d, it + 1, rate, conv, div

        k, _, _, rate, conv, _ = jax.lax.while_loop(
            cond, body, (k_init, jnp.inf, 0, 0.5, False, False))
        return k, rate, conv

    def one_step(t, y, h, lu):
        k1 = rhs(t, y, args)
        k2, r2, c2 = stage(t + _C2 * h, y + h * _A21 * k1, k1, lu, h, y)
        k3, r3, c3 = stage(t + _C3 * h, y + h * (_A31 * k1 + _A32 * k2),
                           _AL31 * k1 + _AL32 * k2, lu, h, y)
        k4, r4, c4 = stage(t + _C4 * h, y + h * (_A41 * k1 + _A42 * k2 + _A43 * k3),
                           _AL41 * k1 + _AL42 * k2 + _AL43 * k3, lu, h, y)
        y1 = y + h * (_B[0] * k1 + _B[1] * k2 + _B[2] * k3 + _B[3] * k4)
        err = h * (_BE[0] * k1 + _BE[1] * k2 + _BE[2] * k3 + _BE[3] * k4)
        rate = jnp.maximum(jnp.maximum(r2, r3), r4)
        conv = c2 & c3 & c4 & jnp.all(jnp.isfinite(y1))
        return y1, wrms(err, y), rate, conv

    ys0 = jnp.zeros((n_save, n), dtype=y0.dtype)

    def _tol(s):
        return 1e-12 * jnp.maximum(1.0, jnp.abs(s))

    # carry: t, y, h_ctrl (unclipped proposal), save_idx, ys, n_steps,
    #        en_prev (last accepted error, for the PI term), dead
    def cond(c):
        t, y, h_ctrl, sidx, ys, nstep, en_prev, dead = c
        return (sidx < n_save) & (~dead) & (nstep < max_steps)

    def body(c):
        t, y, h_ctrl, sidx, ys, nstep, en_prev, dead = c
        next_save = t_eval[sidx]

        def record_only():
            # already at this save time (e.g. t_eval[0] == t0): record, advance,
            # do not take a (zero-width) step.
            return (t, y, h_ctrl, sidx + 1, ys.at[sidx].set(y), nstep, en_prev,
                    dead)

        def do_step():
            # clip the actual step so it never overshoots the next save time
            # (-> recorded exactly at t_eval); the controller still proposes off
            # the unclipped h_ctrl so clipping does not bias the step sequence.
            h = jnp.minimum(h_ctrl, next_save - t)
            J = jac(t, y, args)
            lu = jsla.lu_factor(eye - h * _G * J)
            y1, en, rate, conv = one_step(t, y, h, lu)
            accept = conv & (en <= 1.0)
            # Error controller: an I-term en^(-1/3) with an optional Soderlind PI
            # memory term (en_prev/en)^pi_beta -- the proportional part that damps
            # the controller's overshoot-then-reject oscillation a bare I-controller
            # suffers (pi_beta=0 recovers the pure I-controller). Combined with the
            # Gustafsson-style convergence-rate growth limiter f_conv below, which
            # caps growth by the simplified-Newton contraction rate so a step is not
            # grown into nonlinear divergence.
            ens = jnp.maximum(en, 1e-30)
            f_pi = (en_prev / ens) ** pi_beta
            f_err = safety * jnp.where(en > 0, ens ** (-1.0 / 3.0) * f_pi, maxfac)
            f_conv = jnp.where(rate > 1e-3, theta_target / rate, maxfac)
            fac = jnp.clip(jnp.minimum(f_err, f_conv), minfac, maxfac)
            fac = jnp.where(conv, fac, 0.25)        # nonconvergence -> shrink hard
            h_ctrl_new = h_ctrl * fac
            t_new = jnp.where(accept, t + h, t)
            y_new = jnp.where(accept, y1, y)
            en_prev_new = jnp.where(accept, ens, en_prev)   # update only on accept
            landed = accept & (t_new >= next_save - _tol(next_save))
            ys_new = jax.lax.cond(landed, lambda: ys.at[sidx].set(y_new),
                                  lambda: ys)
            sidx_new = sidx + jnp.where(landed, 1, 0)
            stuck = h_ctrl_new < 1e-13
            return (t_new, y_new, h_ctrl_new, sidx_new, ys_new, nstep + 1,
                    en_prev_new, dead | stuck)

        already = (next_save - t) <= _tol(next_save)
        return jax.lax.cond(already, record_only, do_step)

    init = (jnp.asarray(float(t0)), y0, jnp.asarray(float(h0)), jnp.array(0),
            ys0, jnp.array(0), jnp.array(1.0), jnp.array(False))
    t, y, h, sidx, ys, nstep, en_prev, dead = jax.lax.while_loop(cond, body, init)
    return ys
