"""Cap-free reverse-mode gradients of stiff solves via a hand-written discrete adjoint.

Reverse-mode autodiff *through* a stiff implicit solve (diffrax's
``RecursiveCheckpointAdjoint``) returns non-finite values above an integrator
step-size threshold. The cause is not ill-conditioning -- the per-step operator
``I - dt*J`` stays well-conditioned -- but a floating-point **range overflow in
the reverse accumulation**: the solver stores the stage vector-field values
``f_i ~ ||J||*y`` (large for stiff modes), and reverse-mode AD forms cotangents
of those large stored values, scaled by ``dt`` and compounded across stages and
steps, which exceed the float64 range even though the true gradient is finite.
The usual workaround is a global ``dtmax`` cap that forces tiny steps over the
whole solve.

This module removes the cap by **not differentiating through the solve at all**.
The forward pass is an ordinary (robust, adaptive) diffrax solve; the backward
pass is the *discrete adjoint* written out by hand as a per-step recurrence over
the saved trajectory. For each step the adjoint is a single transposed linear
solve through the same well-conditioned operator the forward step used -- a
contraction, so the cotangent stays bounded and nothing overflows. This is the
classical discrete-adjoint construction for implicit Runge--Kutta methods
(Sandu 2006, *On the Properties of Runge--Kutta Discrete Adjoints*; the forward/
adjoint/tangent integration of FATODE, Zhang & Sandu 2014); it produces the
*exact* gradient of the discrete solve, verified against both finite differences
and the (correct but capped) ``RecursiveCheckpointAdjoint`` gradient.

Two methods are provided. :func:`implicit_euler_adjoint_solve` uses **implicit
Euler** (one implicit stage), for which the discrete adjoint needs only the
saved post-step states; its step map ``y_{n+1} = y_n + dt*f(y_{n+1}, theta)`` has
exact sensitivities

    d y_{n+1} / d y_n    = (I - dt*J)^{-1}
    d y_{n+1} / d theta  = dt*(I - dt*J)^{-1} * df/dtheta,     J = df/dy|_{y_{n+1}}

so the reverse recurrence, given the cotangent ``lam`` of ``y_{n+1}``, is
``mu = (I - dt*J^T)^{-1} lam``; ``theta_bar += dt (df/dtheta)^T mu``; ``lam <-
mu``. Implicit Euler is first order (accurate but many small steps).

:func:`esdirk_adjoint_solve` uses a **high-order ESDIRK** forward (default
``Kvaerno5``, the method the reactors use), whose discrete adjoint recomputes the
stage values in the backward pass and applies the transposed-stage recurrence
(see that function). Either way every per-step solve is the same well-conditioned
``I - gamma*dt*J`` (a contraction), so the cotangents stay bounded with no cap.

**Loss at observation times.** With ``t_eval`` the solve returns the states at
those times, with the discrete adjoint injecting each observation's cotangent at
its step. To keep this exact -- without differentiating through dense
interpolation -- the forward is forced to land steps exactly on ``t_eval`` (via
``diffrax.ClipStepSizeController(step_ts=t_eval)``), so every observation time is
a step boundary. This is what lets the adjoint serve a trajectory loss (the
calibration case), not just a final-state loss.
"""

from __future__ import annotations

from typing import Callable, Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix


def _implicit_tols(rtol: float, atol: float):
    """Explicit root-finder tolerances for an implicit diffrax solver.

    The forward solve here pairs an implicit solver with a step-clipping
    controller. Older diffrax (e.g. 0.7.0, the newest that still supports
    Python 3.10) does not treat ``ClipStepSizeController`` as adaptive, so it
    refuses an implicit solver whose root-finder tolerances are unspecified
    (they default to a "use the stepsize tolerance" sentinel it cannot resolve
    for a non-adaptive controller). Setting them explicitly makes the solve
    valid on every diffrax version.
    """
    return diffrax.VeryChord(rtol=rtol, atol=atol, norm=optimistix.max_norm)


def implicit_euler_adjoint_solve(
    rhs: Callable,
    y0: jnp.ndarray,
    params: jnp.ndarray,
    t_span: tuple[float, float],
    t_eval: Optional[jnp.ndarray] = None,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    dt0: float = 1e-6,
    max_steps: int = 200_000,
) -> jnp.ndarray:
    """Integrate over ``t_span`` with a cap-free discrete-adjoint reverse-mode rule.

    The forward pass is a robust adaptive implicit-Euler diffrax solve. The
    custom VJP is the exact discrete adjoint of that solve, evaluated by a
    backward scan of bounded transposed linear solves -- finite for stiff
    networks at any step size, with no ``dtmax`` cap.

    Parameters
    ----------
    rhs : callable
        Vector field ``rhs(t, y, params) -> dy`` (the reactor RHS), differentiable
        in ``y`` and ``params``.
    y0 : jnp.ndarray
        Initial state, shape ``(n,)``.
    params : jnp.ndarray
        Parameter vector, shape ``(n_params,)``.
    t_span : tuple of float
        ``(t0, t1)`` integration interval.
    t_eval : jnp.ndarray, optional
        Observation times at which to return the state. Must be strictly
        ascending and lie in ``(t0, t1]`` (a time equal to ``t0`` returns ``y0``).
        The forward is forced to step exactly onto these times so the adjoint is
        exact. ``None`` (default) returns only the final state.
    rtol, atol : float
        Forward-solve tolerances (the adaptive controller meets these; the
        adjoint is exact for whatever step sequence results).
    dt0 : float
        Initial step.
    max_steps : int
        Maximum number of accepted steps (also the size of the saved trajectory
        the backward scan walks).

    Returns
    -------
    jnp.ndarray
        If ``t_eval is None``, the final state ``y(t1)``, shape ``(n,)``.
        Otherwise the states at ``t_eval``, shape ``(len(t_eval), n)``. Either
        way the result carries the discrete-adjoint VJP w.r.t. ``y0`` and
        ``params``.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    n = y0.shape[0]
    final_only = t_eval is None
    teval = jnp.asarray([t1] if final_only else t_eval, dtype=jnp.result_type(float))

    term = diffrax.ODETerm(lambda t, y, a: rhs(t, y, a))
    # Force the adaptive controller to land steps exactly on the observation
    # times, so each is a step boundary and the adjoint needs no interpolation.
    controller = diffrax.ClipStepSizeController(
        diffrax.PIDController(rtol=rtol, atol=atol), step_ts=teval
    )

    def _forward(y0_, params_):
        return diffrax.diffeqsolve(
            term, diffrax.ImplicitEuler(root_finder=_implicit_tols(rtol, atol)),
            t0, t1, dt0, y0_, args=params_,
            stepsize_controller=controller,
            saveat=diffrax.SaveAt(steps=True), max_steps=max_steps,
        )

    def _extract(sol, y0_):
        # Combined trajectory [y0, step states] at times [t0, step times]; each
        # t_eval lands exactly on a boundary, so searchsorted gives the index.
        t_all = jnp.concatenate([jnp.array([t0], dtype=sol.ts.dtype), sol.ts])
        y_all = jnp.concatenate([y0_[None, :], sol.ys], axis=0)
        idx = jnp.searchsorted(t_all, teval, side="left")
        return y_all[idx], idx

    @jax.custom_vjp
    def solve(y0_, params_):
        return _extract(_forward(y0_, params_), y0_)[0]

    def solve_fwd(y0_, params_):
        sol = _forward(y0_, params_)
        ys_eval, idx = _extract(sol, y0_)
        return ys_eval, (sol.ts, sol.ys, y0_, params_, idx)

    def solve_bwd(res, ybar):
        ts, ys, y0_, params_, idx = res          # ybar: (n_obs, n)
        K = ts.shape[0]
        valid = jnp.isfinite(ts)
        t_prev = jnp.concatenate([jnp.array([t0], dtype=ts.dtype), ts[:-1]])
        dts = ts - t_prev

        # Distribute observation cotangents: idx==0 -> y0 directly; idx==m>=1 ->
        # the state produced by step m-1.
        injected = jnp.zeros((K, n), dtype=ys.dtype)
        ybar0_obs = jnp.zeros((n,), dtype=ys.dtype)
        n_obs = teval.shape[0]
        for i in range(n_obs):
            step_idx = jnp.maximum(idx[i] - 1, 0)
            injected = injected.at[step_idx].add(
                jnp.where(idx[i] >= 1, ybar[i], 0.0)
            )
            ybar0_obs = ybar0_obs + jnp.where(idx[i] == 0, ybar[i], 0.0)

        def body(lam_pbar, k):
            lam, pbar = lam_pbar
            y_new = ys[k]
            dt = dts[k]
            ok = valid[k] & (dt > 0)
            lam_k = lam + injected[k]            # add this step's observation cotangent
            Jf = jax.jacfwd(lambda y: rhs(0.0, y, params_))(y_new)
            M = jnp.eye(n, dtype=y_new.dtype) - dt * Jf
            mu = jnp.linalg.solve(M.T, lam_k)    # (I - dt*J^T)^{-1} lam_k
            _, vjp = jax.vjp(lambda q: rhs(0.0, y_new, q), params_)
            dpar = dt * vjp(mu)[0]
            lam_new = jnp.where(ok, mu, lam_k)
            pbar_new = jnp.where(ok, pbar + dpar, pbar)
            return (lam_new, pbar_new), None

        (lam0, pbar), _ = jax.lax.scan(
            body, (jnp.zeros((n,), dtype=ys.dtype), jnp.zeros_like(params_)),
            jnp.arange(K - 1, -1, -1),
        )
        return lam0 + ybar0_obs, pbar

    solve.defvjp(solve_fwd, solve_bwd)
    out = solve(y0, params)
    return out[0] if final_only else out


# --- High-order ESDIRK discrete adjoint --------------------------------------
#
# The implicit-Euler adjoint above is first order. For accuracy parity with the
# reactors -- which integrate with the high-order ESDIRK ``Kvaerno5`` -- the same
# idea (robust diffrax forward + hand-written discrete adjoint) extends to a
# general s-stage ESDIRK, at the cost of recomputing the stage values in the
# backward pass and applying the transposed-stage recurrence.
#
# An ESDIRK step over ``dt`` (autonomous f; our reaction RHS ignores t) is
#
#     Y_i = y_n + dt sum_{j<=i} A[i,j] f(Y_j),   k_i = f(Y_i),   i = 0..s-1
#     y_{n+1} = y_n + dt sum_i b_i k_i
#
# with A lower-triangular, A[0,0]=0 (explicit first stage) and A[i,i]=gamma. Its
# discrete adjoint, given the cotangent ``lam`` of y_{n+1}, sweeps the stages in
# reverse (i = s-1..0):
#
#     rhs_i  = dt b_i lam + dt sum_{j>i} A[j,i] Ybar_j
#     Ybar_i = (I - dt*gamma_i J_i^T)^{-1} J_i^T rhs_i          (bounded solve)
#     kappa_i = rhs_i + dt*gamma_i Ybar_i
#     theta_bar += (df/dtheta|Y_i)^T kappa_i
#     lam_n = lam + sum_i Ybar_i
#
# where J_i = df/dy|_{Y_i}. Each diagonal block ``I - dt*gamma_i J_i^T`` is the
# same well-conditioned (contractive) operator the forward stage inverts, so the
# cotangents stay bounded -- finite with no cap. For s=1, gamma=1 (implicit
# Euler) this reduces to lam_n = (I - dt J^T)^{-1} lam, matching the function
# above. (Sandu 2006; FATODE, Zhang & Sandu 2014.)


def _esdirk_tableau(solver):
    """Extract (A, b, gamma_diag, n_stages) as JAX/numpy arrays from a diffrax
    ESDIRK solver's Butcher tableau (the full lower-triangular A with diagonal)."""
    t = solver.tableau
    s = int(t.num_stages)
    A = np.zeros((s, s))
    diag = np.asarray(t.a_diagonal, dtype=float)
    for i in range(s):
        A[i, i] = diag[i]
    for k, row in enumerate(t.a_lower):       # a_lower[k] is row k+1's sub-diagonal
        A[k + 1, : k + 1] = np.asarray(row, dtype=float)
    return jnp.asarray(A), jnp.asarray(np.asarray(t.b_sol, dtype=float)), diag, s


def esdirk_adjoint_solve(
    rhs: Callable,
    y0: jnp.ndarray,
    params: jnp.ndarray,
    t_span: tuple[float, float],
    t_eval: Optional[jnp.ndarray] = None,
    *,
    solver: Optional[diffrax.AbstractSolver] = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    dt0: float = 1e-6,
    max_steps: int = 200_000,
    newton_iters: int = 12,
) -> jnp.ndarray:
    """Cap-free reverse-mode gradient through a high-order ESDIRK solve.

    Like :func:`implicit_euler_adjoint_solve` but the forward uses a high-order
    ESDIRK method (default :class:`diffrax.Kvaerno5`, matching the reactors), and
    the backward is the transposed-stage discrete adjoint of that method. The
    stage values are recomputed in the backward pass (diffrax saves only the
    step states), then the per-stage transposed solves accumulate the gradient.
    Finite for stiff networks with no ``dtmax`` cap; the autonomous reaction RHS
    is assumed (the stage times ``c`` do not enter).

    Parameters
    ----------
    rhs, y0, params, t_span, t_eval, rtol, atol, dt0, max_steps
        As for :func:`implicit_euler_adjoint_solve`.
    solver : diffrax.AbstractSolver, optional
        The ESDIRK forward solver; must expose a Butcher ``tableau``. Defaults to
        :class:`diffrax.Kvaerno5`.
    newton_iters : int, optional
        Newton iterations used to recompute each implicit stage in the backward
        pass. The default converges the well-conditioned stage equation to
        machine precision for the step sizes the adaptive forward selects.

    Returns
    -------
    jnp.ndarray
        Final state ``(n,)`` if ``t_eval is None``, else states at ``t_eval``
        ``(len(t_eval), n)``; carries the discrete-adjoint VJP w.r.t. ``y0`` and
        ``params``.
    """
    if solver is None:
        solver = diffrax.Kvaerno5()
    # Set explicit root-finder tolerances (see _implicit_tols) so the implicit
    # solve is valid on diffrax versions that don't treat ClipStepSizeController
    # as adaptive (older diffrax on Python 3.10).
    solver = eqx.tree_at(
        lambda s: s.root_finder, solver, _implicit_tols(rtol, atol)
    )
    A, b, diag_np, s = _esdirk_tableau(solver)
    diag = jnp.asarray(diag_np)

    t0, t1 = float(t_span[0]), float(t_span[1])
    n = y0.shape[0]
    final_only = t_eval is None
    teval = jnp.asarray([t1] if final_only else t_eval, dtype=jnp.result_type(float))

    term = diffrax.ODETerm(lambda t, y, a: rhs(t, y, a))
    controller = diffrax.ClipStepSizeController(
        diffrax.PIDController(rtol=rtol, atol=atol), step_ts=teval
    )

    def _forward(y0_, params_):
        return diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0_, args=params_,
            stepsize_controller=controller,
            saveat=diffrax.SaveAt(steps=True), max_steps=max_steps,
        )

    def _extract(sol, y0_):
        t_all = jnp.concatenate([jnp.array([t0], dtype=sol.ts.dtype), sol.ts])
        y_all = jnp.concatenate([y0_[None, :], sol.ys], axis=0)
        idx = jnp.searchsorted(t_all, teval, side="left")
        return y_all[idx], idx

    def _stages(y_n, params_, dt):
        # Recompute the ESDIRK stage values Y_i by solving each stage equation
        # Y_i = pred_i + dt*gamma_i f(Y_i) (explicit first stage solves trivially).
        f = lambda y: rhs(0.0, y, params_)
        ks, Ys = [], []
        for i in range(s):
            pred = y_n
            for j in range(i):
                pred = pred + dt * A[i, j] * ks[j]
            Yi = pred
            if diag_np[i] != 0.0:
                gi = diag[i]
                def newton(Y, _):
                    G = Y - pred - dt * gi * f(Y)
                    J = jax.jacfwd(f)(Y)
                    return Y - jnp.linalg.solve(jnp.eye(n) - dt * gi * J, G), None
                Yi, _ = jax.lax.scan(newton, Yi, None, length=newton_iters)
            Ys.append(Yi)
            ks.append(f(Yi))
        return jnp.stack(Ys)                      # (s, n)

    def _step_adjoint(y_n, params_, dt, lam):
        Ys = _stages(y_n, params_, dt)
        Js = jax.vmap(lambda Y: jax.jacfwd(lambda y: rhs(0.0, y, params_))(Y))(Ys)
        Ybar = [None] * s
        pbar = jnp.zeros_like(params_)
        for i in range(s - 1, -1, -1):
            rhs_i = dt * b[i] * lam
            for j in range(i + 1, s):
                rhs_i = rhs_i + dt * A[j, i] * Ybar[j]
            Ji = Js[i]
            M = jnp.eye(n) - dt * diag[i] * Ji.T
            Ybar_i = jnp.linalg.solve(M, Ji.T @ rhs_i)
            Ybar[i] = Ybar_i
            kappa = rhs_i + dt * diag[i] * Ybar_i
            _, vjp = jax.vjp(lambda q: rhs(0.0, Ys[i], q), params_)
            pbar = pbar + vjp(kappa)[0]
        lam_n = lam + sum(Ybar)
        return lam_n, pbar

    @jax.custom_vjp
    def solve(y0_, params_):
        return _extract(_forward(y0_, params_), y0_)[0]

    def solve_fwd(y0_, params_):
        sol = _forward(y0_, params_)
        ys_eval, idx = _extract(sol, y0_)
        return ys_eval, (sol.ts, sol.ys, y0_, params_, idx)

    def solve_bwd(res, ybar):
        ts, ys, y0_, params_, idx = res
        K = ts.shape[0]
        valid = jnp.isfinite(ts)
        t_prev = jnp.concatenate([jnp.array([t0], dtype=ts.dtype), ts[:-1]])
        dts = ts - t_prev
        y_prev = jnp.concatenate([y0_[None, :], ys[:-1]], axis=0)

        injected = jnp.zeros((K, n), dtype=ys.dtype)
        ybar0_obs = jnp.zeros((n,), dtype=ys.dtype)
        for i in range(teval.shape[0]):
            step_idx = jnp.maximum(idx[i] - 1, 0)
            injected = injected.at[step_idx].add(jnp.where(idx[i] >= 1, ybar[i], 0.0))
            ybar0_obs = ybar0_obs + jnp.where(idx[i] == 0, ybar[i], 0.0)

        def body(lam_pbar, k):
            lam, pbar = lam_pbar
            ok = valid[k] & (dts[k] > 0)
            lam_k = lam + injected[k]
            # Skip the (expensive 7-stage) per-step adjoint on padded/invalid
            # slots via cond, so the backward cost scales with the actual step
            # count rather than the allocated buffer -- ``max_steps`` can then be
            # set generously for forward robustness without slowing the backward.
            def do(_):
                lam_n, dpar = _step_adjoint(y_prev[k], params_, dts[k], lam_k)
                return lam_n, pbar + dpar
            lam_new, pbar_new = jax.lax.cond(ok, do, lambda _: (lam_k, pbar), None)
            return (lam_new, pbar_new), None

        (lam0, pbar), _ = jax.lax.scan(
            body, (jnp.zeros((n,), dtype=ys.dtype), jnp.zeros_like(params_)),
            jnp.arange(K - 1, -1, -1),
        )
        return lam0 + ybar0_obs, pbar

    solve.defvjp(solve_fwd, solve_bwd)
    out = solve(y0, params)
    return out[0] if final_only else out
