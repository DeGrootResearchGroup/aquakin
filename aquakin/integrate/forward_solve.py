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

The run is faster than the colored ``diffrax`` forward path -- measured ~0.86x
(compile+run) and ~0.88x (run-only) on the full 609-day BSM2 dynamic solve at the
benchmark hourly save grid -- because the lean loop strips the differentiability
scaffolding while the **dense-output continuous extension** (see below) removes
the only remaining per-save penalty. The per-step costs are otherwise the same as
``diffrax`` (both build the colored Jacobian once per step, which XLA fuses into
one batched JVP -- so freezing/reusing it does not help and in fact hurts via
degraded Newton convergence). The compile is also faster, though the margin is
**scale-dependent**: large for small/simple networks (where the ``diffrax``
implicit scaffolding dominates the trace) and smaller on a big flowsheet (where
the shared colored-Jacobian + plant-RHS tracing dominates). The trade-off versus
``diffrax`` is the loss of differentiability, so a large *differentiable* solve
(``calibrate`` / ``sensitivity`` / ``jax.grad``) must still use the
``colored_jacobian`` ``diffrax`` path. Results match the ``diffrax`` trajectory to
the same ``rtol`` (the usual step-sequence variation between two valid adaptive
solves).

Output at ``t_eval`` uses a **cubic-Hermite dense output** (the Kvaerno3
continuous extension, :func:`_hermite`): the integrator takes its natural adaptive
steps -- clipped only to the final time ``t1`` -- and every save point falling
inside a step is recovered by interpolation, exactly as the ``diffrax`` path does.
This replaces the earlier step-clipping (a forced step boundary on every save),
whose cost grew with save density and was the dominant reason an earlier version
ran slower than ``diffrax`` at the dense benchmark grid; with dense output the run
time is flat in the number of save points.

This is the **conceptual exception** to the single-source-of-truth solver rule:
unlike every other solve path it does not build a ``diffrax`` solver/controller
(it is a hand-rolled ``lax.while_loop`` precisely to escape that machinery), so it
cannot route through ``build_implicit_solver`` / ``build_step_controller``. Its
Kvaerno3 tableau must therefore be kept consistent with diffrax's ``Kvaerno3`` by
hand. Because it constructs no diffrax solver *object*, the single-source guard in
``tests/unit/test_solver_config_single_source.py`` does not flag it (and it needs
no allowlist entry there).

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
earlier ``maxfac=5`` pure-I controller (fewer error-test rejections).

The simplified-Newton inner tolerance ``_KAPPA = 1e-1`` was likewise tuned on that
run: loosening it from ``1e-2`` cut the Newton iterations per stage ~3.7 -> ~3.3
(and the step count fell slightly too, so it is a pure win, not a rejection
trade-off), ~10% faster, while the solution stays within ``rtol`` -- the
final-state difference from the ``diffrax`` solve is ~2.5e-4, far inside the
benchmark agreement. The RHS evaluation (recycle + kinetics), at ~12 per step, is
the dominant cost; the per-step Jacobian factorization is only ~8% (so block /
sparse factorization is not worth it at this scale -- it loses to XLA's dense LU).
"""

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
_KAPPA = 1e-1          # simplified-Newton convergence tolerance (Hairer eta test)


def _hermite(y0_, y1_, f0_, f1_, h, theta):
    """Cubic Hermite continuous extension on a step ``[t, t+h]`` at fraction
    ``theta`` in ``[0, 1]``. Matches the state and its derivative at both ends
    (3rd-order, the order of the Kvaerno3 method and the same dense output the
    ``diffrax`` path uses). ``f0 = rhs(t, y) = k1`` and ``f1 = rhs(t+h, y1) = k4``
    (the stiffly-accurate last stage is evaluated at the step endpoint, so its
    derivative is the endpoint slope -- no extra RHS evaluation needed)."""
    th2 = theta * theta
    th3 = th2 * theta
    h00 = 2.0 * th3 - 3.0 * th2 + 1.0
    h10 = th3 - 2.0 * th2 + theta
    h01 = -2.0 * th3 + 3.0 * th2
    h11 = th3 - th2
    return h00 * y0_ + h10 * h * f0_ + h01 * y1_ + h11 * h * f1_


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
        # k1 (slope at t) and k4 (slope at t+h, the stiffly-accurate endpoint
        # stage) feed the cubic-Hermite dense output.
        return y1, wrms(err, y), rate, conv, k1, k4

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
            # Take the NATURAL adaptive step, clipped only to the final time t1
            # (not to each save). Save points that fall inside the step are
            # recovered by the cubic-Hermite continuous extension below, so the
            # controller keeps its natural step rhythm instead of being forced to
            # plant an extra boundary on every t_eval point (the step-clipping
            # cost at a dense save grid -- issue #386).
            h = jnp.minimum(h_ctrl, t1 - t)
            J = jac(t, y, args)
            lu = jsla.lu_factor(eye - h * _G * J)
            y1, en, rate, conv, k1, k4 = one_step(t, y, h, lu)
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
            # Dense output: on an accepted step, record EVERY save time in
            # (t, t_new] by interpolating within the step (a step may span several
            # saves at a sparse grid, or none at a dense one). f0=k1, f1=k4.
            def rec_cond(rc):
                ys_, si_ = rc
                in_range = si_ < n_save
                tsi = t_eval[jnp.minimum(si_, n_save - 1)]
                return accept & in_range & (tsi <= t_new + _tol(t_new))

            def rec_body(rc):
                ys_, si_ = rc
                theta = (t_eval[si_] - t) / h
                ysave = _hermite(y, y_new, k1, k4, h, theta)
                return ys_.at[si_].set(ysave), si_ + 1

            ys_new, sidx_new = jax.lax.while_loop(rec_cond, rec_body, (ys, sidx))
            stuck = h_ctrl_new < 1e-13
            return (t_new, y_new, h_ctrl_new, sidx_new, ys_new, nstep + 1,
                    en_prev_new, dead | stuck)

        already = (next_save - t) <= _tol(next_save)
        return jax.lax.cond(already, record_only, do_step)

    init = (jnp.asarray(float(t0)), y0, jnp.asarray(float(h0)), jnp.array(0),
            ys0, jnp.array(0), jnp.array(1.0), jnp.array(False))
    t, y, h, sidx, ys, nstep, en_prev, dead = jax.lax.while_loop(cond, body, init)
    return ys
