"""Pseudo-transient continuation (PTC) steady-state solver for plants.

The plant steady state is the root of the flat right-hand side ``F(y) = dy/dt``.
A plain Newton root-find on ``F(y) = 0`` stalls on stiff plants (long-SRT
digesters, slow biomass) -- far from the solution it overshoots into nonphysical
states or stagnates at local minima of ``||F||``. Forward integration
(:meth:`Plant.run_to_steady_state`) is robust but slow and warm-start dependent.

**Pseudo-transient continuation** bridges the two (Kelley & Keyes 1998). It takes
damped-Newton steps

    (V/dt - J) dy = F(y),    y <- y + dy,    J = dF/dy,

where ``V`` is a positive diagonal scaling. The ``V/dt`` term is a pseudo-time
regularisation: at small ``dt`` the step is a stable backward-Euler move along
the physical transient (globally convergent, like time-stepping), and as ``dt``
grows the term vanishes and the step becomes Newton's method (quadratic terminal
convergence). The pseudo-timestep is ramped by Switched-Evolution-Relaxation
(SER), ``dt <- dt * ||F_old|| / ||F_new||`` capped from above, so it grows as the
residual falls. This is the standard robust method for "forward integration
converges but Newton stalls" systems and the basis of pseudo-transient
flowsheet convergence (Pattison & Baldea 2014).

``V = diag(max(|y|, floor))`` (per-state pseudo-time) is essential here: plant
states span orders of magnitude (dissolved O2 ~ 2, heterotrophs ~ 2000, gas
fractions ~ 1e-3), and a scalar ``I/dt`` thrashes; the per-state scaling gives
every state a magnitude-consistent pseudo-time and converges smoothly.

The exact Jacobian ``J`` comes from forward-mode AD of the plant RHS. Gradients
of the steady state with respect to parameters (for design sweeps) are obtained
by the implicit function theorem -- differentiating *through* the iteration is
neither possible (a ``while_loop``) nor necessary, since at the root
``F(y*, theta) = 0`` fixes ``dy*/dtheta = -(dF/dy)^{-1} (dF/dtheta)``.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass
class PTCResult:
    """Outcome of a pseudo-transient continuation steady-state solve.

    Attributes
    ----------
    state : jnp.ndarray
        The steady-state vector ``y*`` (root of the RHS), shape
        ``(n_states,)``. Carries the implicit-function-theorem parameter
        gradient (see module docstring), so ``jax.grad`` through a loss on it
        flows to the plant parameters.
    residual : jnp.ndarray
        The final scaled residual ``max_i |F_i| / max(|y_i|, floor)`` (scalar).
    iterations : jnp.ndarray
        Number of PTC iterations taken (scalar int).
    converged : jnp.ndarray
        Whether ``residual <= tol`` was reached within ``max_iter`` (scalar
        bool). Eager callers may ``bool(...)`` it.
    """

    state: jnp.ndarray
    residual: jnp.ndarray
    iterations: jnp.ndarray
    converged: jnp.ndarray


def _scaled_resnorm(F: jnp.ndarray, y: jnp.ndarray, floor: float) -> jnp.ndarray:
    """Infinity norm of the *relative* rate ``F / max(|y|, floor)``.

    Scaling by the state magnitude weighs every component comparably -- without
    it the solver only ever sees the large biomass modes and the small soluble
    modes never converge.
    """
    return jnp.max(jnp.abs(F) / jnp.maximum(jnp.abs(y), floor))


def ptc_forward(
    rhs: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
    y0: jnp.ndarray,
    *,
    dt0: float = 1e-2,
    dt_max: float = 1e10,
    growth_cap: float = 10.0,
    shrink_floor: float = 0.2,
    max_iter: int = 400,
    tol: float = 1e-6,
    scale_floor: float = 1.0,
    nonneg: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the PTC iteration to a steady state of ``rhs(y, params) = 0``.

    Pure forward solve (no gradient). Implemented as a ``jax.lax.while_loop`` so
    it is jit-compatible; differentiate via :func:`solve_steady_state`, which
    attaches the implicit-function-theorem gradient.

    Parameters
    ----------
    rhs : callable
        ``rhs(y, params) -> dydt`` -- the flat plant right-hand side at a fixed
        (constant) influent.
    params : jnp.ndarray
        Parameter vector passed to ``rhs``.
    y0 : jnp.ndarray
        Initial guess / warm start, shape ``(n_states,)``.
    dt0 : float
        Initial pseudo-timestep. Small is safe (the iteration starts as stable
        time-stepping); the SER ramp grows it automatically.
    dt_max : float
        Cap on the pseudo-timestep (the near-Newton regime).
    growth_cap : float
        Maximum per-step growth factor of the SER ramp (SER is otherwise
        aggressive). ~10 converges the BSM plants smoothly.
    shrink_floor : float
        Minimum per-step factor (a residual increase shrinks ``dt`` toward a
        more stable step), bounded below by this.
    max_iter : int
        Iteration cap.
    tol : float
        Convergence tolerance on the scaled residual.
    scale_floor : float
        Lower bound on ``|y|`` in the per-state scaling (the ``V`` diagonal and
        the residual scaling), so near-zero states do not blow up the scale.
    nonneg : bool
        Clamp the state to ``>= 0`` after each step (concentrations are
        non-negative). Static.

    Returns
    -------
    (state, residual, iterations, converged)
        All ``jnp`` scalars/arrays (trace-safe).
    """
    n = y0.shape[0]
    eye = jnp.eye(n)

    def F(y):
        return rhs(y, params)

    jac = jax.jacfwd(F)

    def step(carry):
        y, dt, r, k = carry
        Fy = F(y)
        J = jac(y)
        s = jnp.maximum(jnp.abs(y), scale_floor)
        # (V/dt - J) dy = F, with V = diag(s): backward-Euler pseudo-time step.
        A = eye * (s / dt)[:, None] - J
        dy = jnp.linalg.solve(A, Fy)
        y_new = y + dy
        if nonneg:
            y_new = jnp.maximum(y_new, 0.0)
        r_new = _scaled_resnorm(F(y_new), y_new, scale_floor)
        # Switched-Evolution-Relaxation: grow dt inversely to the residual,
        # capped above (and floored below) so a residual spike pulls dt back to a
        # stabler step rather than diverging.
        ratio = jnp.clip(r / jnp.maximum(r_new, 1e-30), shrink_floor, growth_cap)
        dt_new = jnp.minimum(dt_max, dt * ratio)
        return (y_new, dt_new, r_new, k + 1)

    def cond(carry):
        _, _, r, k = carry
        return (r > tol) & (k < max_iter)

    r0 = _scaled_resnorm(F(y0), y0, scale_floor)
    y_star, _, r_star, k_star = jax.lax.while_loop(
        cond, step, (y0, jnp.asarray(dt0), r0, jnp.asarray(0))
    )
    converged = r_star <= tol
    return y_star, r_star, k_star, converged


def solve_steady_state(
    rhs: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
    y0: jnp.ndarray,
    **ptc_kwargs,
) -> PTCResult:
    """PTC steady-state solve with implicit-function-theorem parameter gradients.

    Runs :func:`ptc_forward` (its iteration is gradient-blocked), then re-attaches
    the parameter gradient analytically: at the root ``F(y*, params) = 0`` the
    implicit function theorem gives ``dy*/dparams = -(dF/dy)^{-1}(dF/dparams)``,
    so a reverse-mode cotangent ``g`` on the state maps to the parameter gradient
    ``-(dF/dparams)^T (dF/dy)^{-T} g`` -- one transposed Jacobian solve plus one
    vector-Jacobian product, no differentiation through the iteration.

    Parameters
    ----------
    rhs : callable
        ``rhs(y, params) -> dydt``.
    params : jnp.ndarray
        Parameter vector (the differentiable input).
    y0 : jnp.ndarray
        Warm start.
    **ptc_kwargs
        Forwarded to :func:`ptc_forward` (``dt0``, ``growth_cap``, ``tol``, ...).

    Returns
    -------
    PTCResult
        ``state`` carries the IFT gradient w.r.t. ``params``.
    """
    y_star, residual, iterations, converged = ptc_forward(
        rhs, params, y0, **ptc_kwargs
    )
    # The iteration is not reverse-differentiable (a while_loop); block any
    # attempt to differentiate through it and re-inject the exact parameter
    # gradient via the implicit function theorem below.
    y_star = jax.lax.stop_gradient(y_star)
    state = _ift_state(rhs, y_star, params)
    return PTCResult(
        state=state, residual=residual, iterations=iterations, converged=converged
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ift_state(rhs, y_star, params):
    """Identity in ``y_star``; defines the parameter gradient by the IFT.

    Forward returns the converged ``y_star`` unchanged. The custom backward pass
    ignores the (already-converged) ``y_star`` input direction and routes the
    full state cotangent to ``params`` through the implicit function theorem.
    """
    return y_star


def _ift_state_fwd(rhs, y_star, params):
    return y_star, (y_star, params)


def _ift_state_bwd(rhs, res, g):
    y_star, params = res
    # Transposed steady-state Jacobian solve: w = (dF/dy)^{-T} g, using a
    # least-squares solve so a rank-deficient Jacobian (dormant species give
    # zero rows) is handled gracefully.
    J = jax.jacfwd(lambda y: rhs(y, params))(y_star)
    w, *_ = jnp.linalg.lstsq(J.T, g, rcond=None)
    # grad_params = -(dF/dparams)^T w
    _, vjp_params = jax.vjp(lambda p: rhs(y_star, p), params)
    (grad_params,) = vjp_params(-w)
    # No cotangent flows back to the y_star input (it is the fixed root).
    return (jnp.zeros_like(y_star), grad_params)


_ift_state.defvjp(_ift_state_fwd, _ift_state_bwd)
