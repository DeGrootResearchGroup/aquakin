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

This implementation uses the **implicit Euler** method (one implicit stage), for
which the discrete adjoint needs only the saved post-step states. Its step map is
``y_{n+1} = y_n + dt*f(y_{n+1}, theta)``, with exact sensitivities

    d y_{n+1} / d y_n    = (I - dt*J)^{-1}
    d y_{n+1} / d theta  = dt*(I - dt*J)^{-1} * df/dtheta,     J = df/dy|_{y_{n+1}}

so the reverse recurrence, given the cotangent ``lam`` of ``y_{n+1}``, is

    mu        = (I - dt*J^T)^{-1} lam      # bounded transposed solve
    theta_bar += dt * (df/dtheta)^T mu     # parameter-gradient contribution
    lam       <- mu                        # cotangent of y_n

Implicit Euler is first order, so it resolves a given accuracy with more (small)
adaptive steps than a high-order method would; the per-step adjoint stays bounded
because the adaptive forward keeps each step's ``dt*||J||`` moderate. Extending
the same backward scan to a higher-order SDIRK/ESDIRK (recomputing the stage
values in the backward pass and applying the transposed stage tableau) is the
natural next step for efficiency.

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
import jax
import jax.numpy as jnp


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
            term, diffrax.ImplicitEuler(), t0, t1, dt0, y0_, args=params_,
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
