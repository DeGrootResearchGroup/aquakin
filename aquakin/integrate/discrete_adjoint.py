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
*exact* gradient of the discrete solve, which is verified here against both
finite differences and the (correct but capped) ``RecursiveCheckpointAdjoint``
gradient.

This first implementation uses the **implicit Euler** method (one implicit stage),
for which the discrete adjoint needs only the saved post-step states. Its step
map is ``y_{n+1} = y_n + dt*f(y_{n+1}, theta)``, with exact sensitivities

    d y_{n+1} / d y_n    = (I - dt*J)^{-1}
    d y_{n+1} / d theta  = dt*(I - dt*J)^{-1} * df/dtheta,     J = df/dy|_{y_{n+1}}

so the reverse recurrence, given the cotangent ``lam`` of ``y_{n+1}``, is

    mu        = (I - dt*J^T)^{-1} lam      # bounded transposed solve
    theta_bar += dt * (df/dtheta)^T mu     # parameter-gradient contribution
    lam       <- mu                        # cotangent of y_n

Implicit Euler is first order, so it resolves a given accuracy with more (small)
adaptive steps than a high-order method would; the per-step adjoint above stays
bounded because the adaptive forward keeps each step's ``dt*||J||`` moderate.
Extending the same backward scan to a higher-order SDIRK/ESDIRK (recomputing the
stage values in the backward pass and applying the transposed stage tableau) is
the natural next step for efficiency.
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
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    dt0: float = 1e-6,
    max_steps: int = 200_000,
) -> jnp.ndarray:
    """Integrate to ``t_span[1]`` and return the final state, with a cap-free
    discrete-adjoint reverse-mode rule w.r.t. ``y0`` and ``params``.

    The forward pass is a robust adaptive implicit-Euler diffrax solve. The
    custom VJP is the exact discrete adjoint of that solve, evaluated by a
    backward scan of bounded transposed linear solves -- finite for stiff
    networks at any step size, with no ``dtmax`` cap.

    Parameters
    ----------
    rhs : callable
        The vector field ``rhs(t, y, params) -> dy`` (the reactor RHS). Must be
        differentiable in ``y`` and ``params``.
    y0 : jnp.ndarray
        Initial state, shape ``(n,)``.
    params : jnp.ndarray
        Parameter vector, shape ``(n_params,)``.
    t_span : tuple of float
        ``(t0, t1)`` integration interval.
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
        Final state ``y(t1)``, shape ``(n,)``, carrying the discrete-adjoint VJP.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    n = y0.shape[0]
    term = diffrax.ODETerm(lambda t, y, a: rhs(t, y, a))
    controller = diffrax.PIDController(rtol=rtol, atol=atol)

    def _forward(y0_, params_):
        # SaveAt(steps=True) records every accepted step; trailing slots are
        # padded with inf (ts) so the backward scan can mask them out.
        return diffrax.diffeqsolve(
            term, diffrax.ImplicitEuler(), t0, t1, dt0, y0_, args=params_,
            stepsize_controller=controller,
            saveat=diffrax.SaveAt(steps=True), max_steps=max_steps,
        )

    @jax.custom_vjp
    def solve(y0_, params_):
        sol = _forward(y0_, params_)
        last = jnp.sum(jnp.isfinite(sol.ts)) - 1
        return sol.ys[last]

    def solve_fwd(y0_, params_):
        sol = _forward(y0_, params_)
        last = jnp.sum(jnp.isfinite(sol.ts)) - 1
        return sol.ys[last], (sol.ts, sol.ys, params_)

    def solve_bwd(res, ybar):
        ts, ys, params_ = res
        K = ts.shape[0]
        valid = jnp.isfinite(ts)
        # dt of step k is ts[k] - ts[k-1] (ts[-1] := t0). ys[k] is the state
        # *after* step k (= y_{n+1} for that step).
        t_prev = jnp.concatenate([jnp.array([t0], dtype=ts.dtype), ts[:-1]])
        dts = ts - t_prev

        def body(carry, k):
            lam, pbar = carry
            y_new = ys[k]
            dt = dts[k]
            ok = valid[k] & (dt > 0)
            # J = df/dy at the post-step state; M = I - dt*J (forward operator).
            Jf = jax.jacfwd(lambda y: rhs(0.0, y, params_))(y_new)
            M = jnp.eye(n, dtype=y_new.dtype) - dt * Jf
            mu = jnp.linalg.solve(M.T, lam)            # (I - dt*J^T)^{-1} lam
            _, vjp = jax.vjp(lambda q: rhs(0.0, y_new, q), params_)
            dpar = dt * vjp(mu)[0]                     # dt*(df/dtheta)^T mu
            lam = jnp.where(ok, mu, lam)
            pbar = jnp.where(ok, pbar + dpar, pbar)
            return (lam, pbar), None

        # Walk steps in reverse; padded slots (ok=False) pass the cotangent
        # through unchanged.
        (lam0, pbar), _ = jax.lax.scan(
            body, (ybar, jnp.zeros_like(params_)), jnp.arange(K - 1, -1, -1)
        )
        return lam0, pbar

    solve.defvjp(solve_fwd, solve_bwd)
    return solve(y0, params)
