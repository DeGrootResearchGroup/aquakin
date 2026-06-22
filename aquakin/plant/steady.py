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

A **step-acceptance guard** makes the ramp robust to a far-from-solution
overshoot: a step whose scaled residual is non-finite or grows past a generous
factor is rejected (the iterate is held and ``dt`` hard-shrunk toward the stable
backward-Euler limit) rather than accepted. PTC is legitimately non-monotone --
a healthy step can spike the residual and recover -- so the threshold is generous
and a converging run never rejects (it is bit-identical to the unguarded
iteration); only a genuine divergence (which would otherwise run to ``NaN``, e.g.
from a cold start) is pulled back.

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
from typing import Callable, Optional

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
    divergence_factor: float = 1000.0,
    jac_fn: Optional[Callable] = None,
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
    divergence_factor : float
        Step-acceptance guard threshold. A step is rejected (the iterate is held
        and ``dt`` hard-shrunk) when the scaled residual is non-finite or grows by
        more than this factor in one step -- catching a far-from-solution Newton
        overshoot that would otherwise run to ``NaN`` (a cold-start divergence).
        PTC is legitimately non-monotone (a healthy step can spike the residual
        ~30x and recover), so the default ``1000`` is deliberately generous: it
        sits in the wide gap between benign and catastrophic growth, so a
        converging run never rejects (it is bit-identical to no guard) while a
        diverging one is pulled back. Set to ``inf`` to reject only non-finite
        steps.
    jac_fn : callable, optional
        Custom Jacobian materializer ``(F, y) -> dF/dy``, used in place of the
        default dense ``jax.jacfwd(F)`` at each iteration. The PTC step
        ``(V/dt - J) dy = F`` is unchanged, so the result is identical whenever
        ``jac_fn`` returns the true Jacobian (e.g. a column-compressed colored-AD
        materializer, which equals the dense Jacobian on its sparsity-pattern
        support but forms it in far fewer Jacobian-vector products). ``None``
        (default) uses dense ``jax.jacfwd``.

    Returns
    -------
    (state, residual, iterations, converged)
        All ``jnp`` scalars/arrays (trace-safe).
    """
    n = y0.shape[0]
    eye = jnp.eye(n)
    reject_shrink = 0.1     # dt multiplier on a rejected step (toward stability)
    dt_min = 1e-12          # floor so a rejected dt cannot underflow to zero

    def F(y):
        return rhs(y, params)

    jac = jax.jacfwd(F) if jac_fn is None else (lambda y: jac_fn(F, y))

    def step(carry):
        y, dt, r, k = carry
        Fy = F(y)
        J = jac(y)
        s = jnp.maximum(jnp.abs(y), scale_floor)
        # (V/dt - J) dy = F, with V = diag(s): backward-Euler pseudo-time step.
        A = eye * (s / dt)[:, None] - J
        dy = jnp.linalg.solve(A, Fy)
        y_trial = y + dy
        if nonneg:
            y_trial = jnp.maximum(y_trial, 0.0)
        r_trial = _scaled_resnorm(F(y_trial), y_trial, scale_floor)
        # Step-acceptance guard: reject a non-finite or grossly-diverging step
        # (residual growth past the generous ``divergence_factor``) -- hold the
        # iterate and hard-shrink dt so the retry is a stabler backward-Euler
        # step, instead of accepting an overshoot that runs to NaN. The threshold
        # sits in the wide gap between PTC's benign non-monotone spikes (~30x) and
        # a true blow-up (~1e5x), so a converging run never rejects and stays
        # bit-identical to the unguarded iteration.
        accept = jnp.isfinite(r_trial) & (r_trial <= divergence_factor * r)
        y_new = jnp.where(accept, y_trial, y)
        r_new = jnp.where(accept, r_trial, r)
        # Accept: Switched-Evolution-Relaxation grows dt inversely to the residual
        # (capped above, floored below). Reject: hard-shrink dt toward the stable
        # limit (floored so it cannot underflow to zero).
        ser = jnp.clip(r / jnp.maximum(r_trial, 1e-30), shrink_floor, growth_cap)
        dt_new = jnp.where(accept,
                           jnp.minimum(dt_max, dt * ser),
                           jnp.maximum(dt * reject_shrink, dt_min))
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
    *,
    jac_fn: Optional[Callable] = None,
    primal_rhs: Optional[Callable] = None,
    **ptc_kwargs,
) -> PTCResult:
    """PTC steady-state solve with implicit-function-theorem parameter gradients.

    Runs :func:`ptc_forward` (its iteration is gradient-blocked), then re-attaches
    the parameter gradient analytically: at the root ``F(y*, params) = 0`` the
    implicit function theorem gives ``dy*/dparams = -(dF/dy)^{-1}(dF/dparams)``,
    so a reverse-mode cotangent ``g`` on the state maps to the parameter gradient
    ``-(dF/dparams)^T (dF/dy)^{-T} g`` -- one transposed Jacobian solve plus one
    vector-Jacobian product, no differentiation through the iteration.

    .. note::
       The gradient assumes a **full-rank** steady Jacobian ``dF/dy`` at the
       root -- the case for every shipped network at its operating point, where
       the transposed solve is exact. If the Jacobian is rank-deficient (a fully
       dormant/depleted species gives a zero row, so the root is not locally
       unique along that null direction), the true IFT cotangent is undefined and
       the backward returns the least-squares min-norm cotangent instead; a
       gradient w.r.t. a parameter that moves only a dormant species is then not
       reliable. The forward iteration is unaffected -- its pseudo-time term
       keeps the per-step operator non-singular regardless.

    Parameters
    ----------
    rhs : callable
        ``rhs(y, params) -> dydt``.
    params : jnp.ndarray
        Parameter vector (the differentiable input).
    y0 : jnp.ndarray
        Warm start.
    jac_fn : callable, optional
        Custom Jacobian materializer ``(F, y) -> dF/dy`` for the PTC iteration
        (e.g. a colored-AD column-compression builder); see :func:`ptc_forward`.
        It accelerates only the iteration Jacobian; the one-shot
        implicit-function-theorem gradient Jacobian below stays dense (a single
        evaluation, negligible against the ~tens of iteration Jacobians).
    primal_rhs : callable, optional
        Alternate ``rhs(y, params)`` for the **forward iteration only** -- e.g.
        one closing over a precomputed (cached) recycle map, which is identical to
        the probed map but cheaper and free of the per-call probing that leaks a
        traced intermediate under ``jit``. The **gradient** (the
        implicit-function-theorem backward) uses the map-*recomputing* ``rhs`` so
        a parameter the recycle map depends on (a flow setpoint) keeps its
        ``d(map)/d(param)`` term. ``None`` (default) uses ``rhs`` for both.
    **ptc_kwargs
        Forwarded to :func:`ptc_forward` (``dt0``, ``growth_cap``, ``tol``, ...).

    Returns
    -------
    PTCResult
        ``state`` carries the IFT gradient w.r.t. ``params``.
    """
    y_star, residual, iterations, converged = ptc_forward(
        rhs if primal_rhs is None else primal_rhs,
        params, y0, jac_fn=jac_fn, **ptc_kwargs
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
    # Transposed steady-state Jacobian solve: w = (dF/dy)^{-T} g.
    #
    # The IFT cotangent w = J^{-T} g is exact and unique only when the steady
    # Jacobian J = dF/dy is FULL RANK -- true for every shipped/validated network
    # at its operating point, where this least-squares solve equals the plain
    # `solve` the forward PTC step uses (so the forward and backward agree). If J
    # is rank-deficient (e.g. a fully dormant/depleted species contributes a zero
    # row, so the root is not locally unique along that null direction), the true
    # IFT cotangent is undefined; `lstsq` then returns the MIN-NORM least-squares
    # cotangent -- a reasonable but arbitrary choice, NOT the exact gradient. So a
    # gradient w.r.t. a parameter that moves only a dormant species is not
    # reliable there. (A Tikhonov shift would merely substitute a different
    # arbitrary regularization while risking the full-rank gradients, so it is not
    # used; see the full-rank note in `solve_steady_state`.)
    J = jax.jacfwd(lambda y: rhs(y, params))(y_star)
    w, *_ = jnp.linalg.lstsq(J.T, g, rcond=None)
    # grad_params = -(dF/dparams)^T w
    _, vjp_params = jax.vjp(lambda p: rhs(y_star, p), params)
    (grad_params,) = vjp_params(-w)
    # No cotangent flows back to the y_star input (it is the fixed root).
    return (jnp.zeros_like(y_star), grad_params)


_ift_state.defvjp(_ift_state_fwd, _ift_state_bwd)
