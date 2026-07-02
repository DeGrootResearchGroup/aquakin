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
    reject_shrink = 0.1  # dt multiplier on a rejected step (toward stability)
    dt_min = 1e-12  # floor so a rejected dt cannot underflow to zero

    def F(y):
        return rhs(y, params)

    jac = jax.jacfwd(F) if jac_fn is None else (lambda y: jac_fn(F, y))

    # Globalization of the PTC step. A bare backward-Euler step at a large
    # pseudo-time can overshoot -- through zero into the post-solve clip, or far
    # past the trajectory -- and, with no control on the residual, wander or
    # compound to a non-physical value that never recovers within ``max_iter``.
    # Two safeguards prevent this:
    #  (1) a positivity-preserving step length (fraction-to-boundary): the step is
    #      capped so a currently-positive state is not overshot through zero, where
    #      the post-solve clip would otherwise distort the step the residual test
    #      sees and seed a divergence; and
    #  (2) a backtracking line search that shrinks the step until its residual is
    #      within ``divergence_factor`` of the BEST residual seen so far. Gating
    #      against the best (not the current) residual is what forbids the runaway:
    #      a permissive bound on the *current* residual lets each accepted spike
    #      raise the ceiling for the next, ratcheting up without limit, whereas the
    #      best residual only decreases, so the bound only tightens -- while still
    #      admitting PTC's benign transient spikes that a strictly monotone line
    #      search would suppress (and which the pseudo-time ramp relies on). This is
    #      deliberately PERMISSIVE (a wide ``divergence_factor``): the cheap
    #      algebraic fallbacks downstream -- parameter continuation and pseudo-
    #      arclength (:func:`continuation_solve`, :func:`arclength_continuation_solve`)
    #      -- recover the out-of-basin / near-fold cases far faster than the
    #      time-integration backstop, so the right design is a wide-basin PTC that
    #      converges most cases quickly and hands the rest to those fallbacks.
    # On a step that cannot be brought within the bound, the iterate is held and dt
    # hard-shrunk for a stabler retry.
    ftb_tau = 0.95  # fraction-to-boundary safety factor (item 1)
    ls_beta = 0.5  # backtracking step factor (item 2)
    ls_max = 20  # max backtracks (alpha shrinks to ~1e-6)

    def step(carry):
        y, dt, r, r_best, k = carry
        Fy = F(y)
        J = jac(y)
        s = jnp.maximum(jnp.abs(y), scale_floor)
        # (V/dt - J) dy = F, with V = diag(s): backward-Euler pseudo-time step.
        A = eye * (s / dt)[:, None] - J
        dy = jnp.linalg.solve(A, Fy)

        # (1) Positivity-preserving max step length: the largest alpha keeping each
        # currently-positive state nonnegative, with a safety factor; 1 when the
        # step does not drive any positive state down (or ``nonneg`` is off).
        if nonneg:
            ratio = jnp.where((dy < 0.0) & (y > 0.0), y / -dy, jnp.inf)
            alpha0 = jnp.minimum(1.0, ftb_tau * jnp.min(ratio))
        else:
            alpha0 = jnp.asarray(1.0)

        def _trial(alpha):
            yt = y + alpha * dy
            if nonneg:
                yt = jnp.maximum(yt, 0.0)  # clip any residual roundoff negatives
            return yt, _scaled_resnorm(F(yt), yt, scale_floor)

        # (2) Backtrack while the step overshoots beyond divergence_factor x the
        # best residual so far (non-monotone bound). A converging step accepts
        # ``alpha0`` with no backtrack.
        bound = divergence_factor * r_best
        yt0, rt0 = _trial(alpha0)

        def _ls_cond(c):
            _, _, rt, j = c
            return (~(jnp.isfinite(rt) & (rt <= bound))) & (j < ls_max)

        def _ls_body(c):
            alpha, _, _, j = c
            a = alpha * ls_beta
            yt, rt = _trial(a)
            return (a, yt, rt, j + 1)

        _, y_trial, r_trial, _ = jax.lax.while_loop(
            _ls_cond, _ls_body, (alpha0, yt0, rt0, jnp.asarray(0))
        )

        # Accept a step within the bound; otherwise hold the iterate and hard-shrink
        # dt for a stabler retry (the net for a direction the line search could not
        # bring within the bound).
        accept = jnp.isfinite(r_trial) & (r_trial <= bound)
        y_new = jnp.where(accept, y_trial, y)
        r_new = jnp.where(accept, r_trial, r)
        r_best_new = jnp.minimum(r_best, r_new)
        # Accept: Switched-Evolution-Relaxation grows dt inversely to the residual
        # ratio (capped above, floored below). Reject: hard-shrink dt toward the
        # stable limit (floored so it cannot underflow to zero).
        ser = jnp.clip(r / jnp.maximum(r_trial, 1e-30), shrink_floor, growth_cap)
        dt_new = jnp.where(
            accept, jnp.minimum(dt_max, dt * ser), jnp.maximum(dt * reject_shrink, dt_min)
        )
        return (y_new, dt_new, r_new, r_best_new, k + 1)

    def cond(carry):
        _, _, r, _, k = carry
        return (r > tol) & (k < max_iter)

    r0 = _scaled_resnorm(F(y0), y0, scale_floor)
    y_star, _, r_star, _, k_star = jax.lax.while_loop(
        cond, step, (y0, jnp.asarray(dt0), r0, r0, jnp.asarray(0))
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
       root -- the case for every shipped model at its operating point, where
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
        rhs if primal_rhs is None else primal_rhs, params, y0, jac_fn=jac_fn, **ptc_kwargs
    )
    # The iteration is not reverse-differentiable (a while_loop); block any
    # attempt to differentiate through it and re-inject the exact parameter
    # gradient via the implicit function theorem below.
    y_star = jax.lax.stop_gradient(y_star)
    state = _ift_state(rhs, y_star, params)
    return PTCResult(state=state, residual=residual, iterations=iterations, converged=converged)


@dataclass
class ContinuationResult:
    """Outcome of a natural-parameter continuation steady-state solve.

    Attributes
    ----------
    state : jnp.ndarray
        Steady state at ``params_target`` (carries the implicit-function-theorem
        parameter gradient, like :func:`solve_steady_state`).
    residual : jnp.ndarray
        Final corrector scaled residual.
    converged : jnp.ndarray
        Whether the path reached ``params_target`` (arc ``s = 1``) with every
        corrector converged.
    continuation_steps : int
        Number of accepted parameter steps taken along the path.
    corrector_iterations : int
        Total PTC iterations summed over every corrector solve.
    """

    state: jnp.ndarray
    residual: jnp.ndarray
    converged: jnp.ndarray
    continuation_steps: int
    corrector_iterations: int


def make_continuation_kernels(rhs, jac_fn=None, ptc_kwargs=None):
    """Build the jitted predictor / corrector for :func:`continuation_solve`.

    The plant ``rhs`` is captured, so the pair compiles once; pass them back via
    ``kernels=`` so a sweep over many target parameter sets (e.g. a sensitivity
    screen) reuses one compilation instead of recompiling per target. The
    predictor and corrector take the parameters as arguments, so a single
    compilation serves every target.
    """
    ptc_kwargs = dict(ptc_kwargs or {})

    @jax.jit
    def predict(params_known, dtheta, y, s, ds):
        # Euler tangent predictor along the path params(s) = params_known + s*dtheta:
        # dy/ds = -(dF/dy)^{-1} (dF/dparams) dparams, extrapolated by one step ds.
        params_s = params_known + s * dtheta
        J = jax.jacfwd(lambda yy: rhs(yy, params_s))(y)
        _, dFds = jax.jvp(lambda th: rhs(y, th), (params_s,), (dtheta,))
        t = jnp.linalg.solve(J, -dFds)
        y_pred = y + ds * t
        # Zeroth-order fallback (hold y) if the tangent is non-finite -- a
        # near-singular dF/dy near a fold, where a small ds and the corrector
        # recover (pseudo-arclength is the rigorous fold treatment, a later step).
        return jnp.where(jnp.all(jnp.isfinite(y_pred)), y_pred, y)

    @jax.jit
    def correct(params_next, y_pred):
        return ptc_forward(rhs, params_next, y_pred, jac_fn=jac_fn, **ptc_kwargs)

    return predict, correct


def continuation_solve(
    rhs,
    params_known,
    y_known,
    params_target,
    *,
    kernels=None,
    jac_fn=None,
    ds0: float = 0.25,
    ds_min: float = 1.0e-3,
    ds_grow: float = 1.7,
    ds_shrink: float = 0.5,
    max_continuation_steps: int = 200,
    **ptc_kwargs,
) -> ContinuationResult:
    """Reach a far steady state by natural-parameter predictor-corrector continuation.

    A direct PTC solve from a warm start fails when the start is outside the basin
    of the target steady state -- the case for a parameter set far from the one the
    warm start was built for. Continuation reaches it by *deforming* from a known
    solution: it steps the parameters along the segment
    ``params(s) = params_known + s (params_target - params_known)``, ``s: 0 -> 1``,
    and at each point an Euler (tangent) predictor supplies the next initial guess
    and :func:`ptc_forward` (the permissive r_best-bounded corrector) refines it. The
    step ``ds`` is adaptive -- it grows on easy convergence and shrinks (the iterate
    held) on a corrector failure -- so every corrector starts inside the basin of
    the branch continuously connected to ``(params_known, y_known)``. This both
    reaches states a cold solve cannot and tracks the *physical* branch (the one
    connected to the known operating point) rather than whatever root a far solve
    lands in (Allgower & Georg, Numerical Continuation Methods).

    Parameters
    ----------
    rhs : callable
        ``rhs(y, params) -> dydt``.
    params_known, y_known : jnp.ndarray
        A known steady state: ``rhs(y_known, params_known) ~ 0``.
    params_target : jnp.ndarray
        The parameter set whose steady state is wanted.
    kernels : tuple, optional
        Pre-built ``(predict, correct)`` from :func:`make_continuation_kernels`,
        to avoid recompiling per target in a sweep. Built internally if ``None``.
    ds0, ds_min, ds_grow, ds_shrink, max_continuation_steps : float / int
        Adaptive continuation-step controls.
    **ptc_kwargs
        Forwarded to the :func:`ptc_forward` corrector.

    Returns
    -------
    ContinuationResult
    """
    if kernels is None:
        predict, correct = make_continuation_kernels(rhs, jac_fn, ptc_kwargs)
    else:
        predict, correct = kernels
    dtheta = params_target - params_known
    y = jax.lax.stop_gradient(y_known)
    s = 0.0
    ds = float(ds0)
    total_iters = 0
    accepted = 0
    residual = jnp.asarray(jnp.inf)
    for _ in range(int(max_continuation_steps)):
        if s >= 1.0 - 1.0e-9:
            break
        ds = min(ds, 1.0 - s)
        y_pred = predict(params_known, dtheta, y, jnp.asarray(s), jnp.asarray(ds))
        params_next = params_known + (s + ds) * dtheta
        y_new, res, iters, conv = correct(params_next, y_pred)
        total_iters += int(iters)
        if bool(conv):
            y = y_new
            s = s + ds
            residual = res
            ds = min(ds * ds_grow, 1.0)
            accepted += 1
        else:
            ds = ds * ds_shrink
            if ds < ds_min:
                break
    converged = bool(s >= 1.0 - 1.0e-9)
    y_star = jax.lax.stop_gradient(y)
    state = _ift_state(rhs, y_star, params_target)
    return ContinuationResult(
        state=state,
        residual=residual,
        converged=jnp.asarray(converged),
        continuation_steps=accepted,
        corrector_iterations=total_iters,
    )


@dataclass
class ArclengthResult:
    """Outcome of a pseudo-arclength continuation steady-state solve.

    Attributes
    ----------
    state : jnp.ndarray
        The operating point at ``params_target`` when ``status == "converged"``;
        otherwise the last point reached on the branch (the fold, for
        ``"past_fold"``). Carries the implicit-function-theorem parameter gradient.
    residual : jnp.ndarray
        Final scaled residual.
    status : str
        ``"converged"`` -- the operating branch reached ``params_target``, so the
        operating point exists (and is returned). ``"past_fold"`` -- the branch
        turned back at a fold (saddle-node bifurcation) *before* the target, so no
        operating-branch steady state exists there: the parameters are past the
        survival limit and the only steady state is on another branch (e.g.
        washout). ``"failed"`` -- the continuation could not advance (inconclusive).
    s_max : float
        Furthest continuation parameter reached. For ``"past_fold"`` this is the
        fold location: the operating branch exists only for ``s <= s_max < 1``.
    continuation_steps, corrector_iterations : int
        Work counters.
    """

    state: jnp.ndarray
    residual: jnp.ndarray
    status: str
    s_max: float
    continuation_steps: int
    corrector_iterations: int

    @property
    def converged(self):
        return jnp.asarray(self.status == "converged")


def make_arclength_kernels(rhs, scale, ptc_kwargs=None):
    """Build the jitted scaled-pseudo-arclength corrector / tangent kernels.

    Works in the scaled state ``z = y / scale`` so the large (biomass) and tiny
    (gas) states contribute comparably to the arclength -- otherwise the arclength
    is swamped by the large states and the continuation parameter ``s`` barely
    advances. The augmented operator ``A = [[F_z, F_s], [t^T]]`` with
    ``F_z = (dF/dy) diag(scale)`` is non-singular even where ``dF/dy`` is singular
    (the fold / soft direction), so the corrector converges to machine precision
    where a plain Newton/PTC step overshoots. The tangent (the predictor and the
    fold detector) is the free AD direction. Captures ``rhs`` and ``scale``, so it
    compiles once; reuse across targets via ``kernels=``.
    """
    ptc_kwargs = dict(ptc_kwargs or {})
    floor = ptc_kwargs.get("scale_floor", 1.0)
    n = int(scale.shape[0])

    @jax.jit
    def corrector(z, s, pk, dtheta, uz, us, tz, ts, dsig):
        y = scale * z
        ps = pk + s * dtheta
        F = rhs(y, ps)
        Fz = jax.jacfwd(lambda yy: rhs(yy, ps))(y) * scale[None, :]
        _, Fs = jax.jvp(lambda th: rhs(y, th), (ps,), (dtheta,))
        N = tz @ (z - uz) + ts * (s - us) - dsig
        A = jnp.zeros((n + 1, n + 1))
        A = A.at[:n, :n].set(Fz).at[:n, n].set(Fs).at[n, :n].set(tz).at[n, n].set(ts)
        du = jnp.linalg.solve(A, -jnp.concatenate([F, jnp.asarray([N])]))
        return z + du[:n], s + du[n], _scaled_resnorm(F, y, floor), A

    @jax.jit
    def tangent(A, tz, ts):
        t = jnp.linalg.solve(A, jnp.zeros(n + 1).at[n].set(1.0))
        t = t / jnp.linalg.norm(t)
        t = jnp.where((t[:n] @ tz + t[n] * ts) < 0.0, -t, t)
        return t[:n], t[n]

    @jax.jit
    def init_tangent(z0, pk, dtheta):
        y = scale * z0
        Fz = jax.jacfwd(lambda yy: rhs(yy, pk))(y) * scale[None, :]
        _, Fs = jax.jvp(lambda th: rhs(y, th), (pk,), (dtheta,))
        t = jnp.concatenate([jnp.linalg.solve(Fz, -Fs), jnp.asarray([1.0])])
        t = t / jnp.linalg.norm(t)
        return t[:n], t[n]

    return corrector, tangent, init_tangent


def arclength_continuation_solve(
    rhs,
    params_known,
    y_known,
    params_target,
    *,
    kernels=None,
    scale=None,
    ptc_kwargs=None,
    dsigma0=0.05,
    dsigma_min=1.0e-5,
    dsigma_max=2.0,
    ds_grow=1.6,
    ds_shrink=0.5,
    max_steps=1500,
    corrector_iters=15,
    tol=1.0e-6,
    nonneg=True,
    s_tol=1.0e-4,
    fold_margin=2.0e-2,
):
    """Reach a far operating point by scaled pseudo-arclength continuation.

    Tracks the steady-state branch ``F(y, s) = rhs(y, params_known + s*dtheta) = 0``
    from the known solution (``s=0``) toward ``params_target`` (``s=1``) by
    arclength, with the fold-regularizing augmented corrector
    (:func:`make_arclength_kernels`). It does two things a direct solve and
    natural-parameter continuation cannot: it reaches operating points behind a
    near-singular ``dF/dy`` (the augmented operator does not invert it), and it
    **detects when the operating branch folds before the target** -- a saddle-node
    bifurcation past which no operating-branch steady state exists (the system can
    only be on another branch, e.g. washout). The ``status`` of the result is the
    physical operating-regime criterion: ``"converged"`` (the operating point
    exists, returned) vs ``"past_fold"`` (it does not -- the sample is outside the
    viable regime). (Allgower & Georg; Keller pseudo-arclength.)

    Returns
    -------
    ArclengthResult
    """
    ptc_kwargs = dict(ptc_kwargs or {})
    tol = float(ptc_kwargs.get("tol", tol))
    if scale is None:
        scale = jnp.maximum(jnp.abs(y_known), 1.0e-3)
    if kernels is None:
        corrector, tangent, init_tangent = make_arclength_kernels(rhs, scale, ptc_kwargs)
    else:
        corrector, tangent, init_tangent = kernels
    dtheta = params_target - params_known
    z = jax.lax.stop_gradient(jnp.asarray(y_known)) / scale
    s = 0.0
    tz, ts = init_tangent(z, params_known, dtheta)
    dsig = float(dsigma0)
    ncorr = [0]
    s_max = 0.0
    last_rF = jnp.asarray(jnp.inf)

    def _correct(z0, s0, uz, us, tzc, tsc, ds):
        zc, sc, A = z0, s0, None
        rF = jnp.asarray(jnp.inf)
        for _ in range(int(corrector_iters)):
            zc, sc, rF, A = corrector(zc, sc, params_known, dtheta, uz, us, tzc, tsc, ds)
            ncorr[0] += 1
            if nonneg:
                zc = jnp.maximum(zc, 0.0)
            if float(rF) < tol:
                return zc, sc, rF, A, True
        return zc, sc, rF, A, False

    def _result(state, res, status, smax, step):
        y_star = jax.lax.stop_gradient(state)
        return ArclengthResult(
            _ift_state(rhs, y_star, params_target), res, status, float(smax), step, ncorr[0]
        )

    for step in range(int(max_steps)):
        ts_prev = float(ts)
        zc = z + dsig * tz
        sc = s + dsig * ts
        if nonneg:
            zc = jnp.maximum(zc, 0.0)
        zc, sc, rF, A, conv = _correct(zc, sc, z, s, tz, ts, dsig)
        if not conv:
            dsig *= ds_shrink
            if dsig < dsigma_min:
                return _result(scale * z, last_rF, "failed", s_max, step)
            continue
        s_new = float(sc)
        # Reachable: the branch crosses s=1 while still advancing forward -- secant
        # on the arclength step to land exactly on the target (the augmented
        # corrector stays well-conditioned, unlike a natural-parameter solve).
        if (s < 1.0 <= s_new) and ts_prev > 0.0:
            d_lo, s_lo, d_hi, s_hi = 0.0, s, dsig, s_new
            zp, rp = zc, rF
            for _ in range(12):
                dt = d_lo + (d_hi - d_lo) * (1.0 - s_lo) / max(s_hi - s_lo, 1e-12)
                zp0 = z + dt * tz
                if nonneg:
                    zp0 = jnp.maximum(zp0, 0.0)
                zp, sp, rp, _, cc = _correct(zp0, s + dt * ts, z, s, tz, ts, dt)
                spf = float(sp)
                if cc and abs(spf - 1.0) < s_tol:
                    return _result(scale * zp, rp, "converged", 1.0, step)
                if spf < 1.0:
                    d_lo, s_lo = dt, spf
                else:
                    d_hi, s_hi = dt, spf
            status = "converged" if float(rp) < tol else "failed"
            return _result(scale * zp, rp, status, max(s_max, 1.0), step)
        z, s = zc, sc
        last_rF = rF
        s_max = max(s_max, s)
        tz, ts = tangent(A, tz, ts)
        # Fold: the s-velocity actually REVERSED (the branch reached its nose and
        # turned back) while short of the target -- a saddle-node bifurcation, so no
        # operating point exists at ``params_target`` (the system is past the
        # survival limit). Only a true sign change qualifies: ``ts`` dips small and
        # *recovers* in high-sensitivity regions of a perfectly reachable branch (a
        # marginally-stable operating point that does exist), so a small-``ts`` test
        # would wrongly exclude real operating points.
        if ts_prev > 0.0 and float(ts) <= 0.0 and s_max < 1.0 - fold_margin:
            return _result(scale * z, rF, "past_fold", s_max, step)
        dsig = min(dsig * ds_grow, dsigma_max)
    return _result(scale * z, last_rF, "failed", s_max, max_steps)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _ift_state(rhs, y_star, params):
    """Identity in ``y_star``; defines the parameter sensitivity by the IFT.

    Returns the converged ``y_star`` unchanged, but attaches the parameter
    sensitivity through the implicit function theorem so the steady state is
    differentiable with respect to ``params`` in **both** AD directions. The
    custom JVP gives the forward-mode sensitivity ``dy = -J^{-1}(dF/dparams)``;
    because that map is *linear in the input tangent*, JAX transposes it
    automatically to recover the reverse-mode gradient ``-(dF/dparams)^T J^{-T}g``
    -- so one rule serves forward (``jax.jvp`` / ``jacfwd``, the many-output
    sensitivity-screen direction) and reverse (``jax.grad`` / ``jacrev``, the
    calibration-gradient direction) alike. The (already-converged) ``y_star``
    input carries no tangent: the root does not depend on the initial guess.

    ``J = dF/dy`` is full rank for every shipped/validated model at its operating
    point, where ``jnp.linalg.solve`` is exact (and equals the plain solve the
    forward PTC step uses, so forward and reverse agree). If ``J`` were
    rank-deficient -- a fully dormant/depleted species contributing a zero row, so
    the root is not locally unique along that null direction -- the IFT sensitivity
    would be undefined, and a sensitivity that moves only such a dormant species is
    not reliable. (A min-norm ``lstsq`` would substitute a different arbitrary
    choice there while risking the full-rank sensitivities, so it is not used.)
    """
    return y_star


@_ift_state.defjvp
def _ift_state_jvp(rhs, primals, tangents):
    y_star, params = primals
    _, dparams = tangents
    # At the steady state F(y*, params) = 0, so J dy + (dF/dparams) dparams = 0
    # (J = dF/dy), giving dy = -J^{-1} (dF/dparams . dparams). The directional
    # parameter derivative dF/dparams . dparams is one forward JVP of the RHS; the
    # input-guess tangent does not enter (the converged root is guess-independent).
    _, dF = jax.jvp(lambda p: rhs(y_star, p), (params,), (dparams,))
    J = jax.jacfwd(lambda y: rhs(y, params))(y_star)
    dy = -jnp.linalg.solve(J, dF)
    return y_star, dy
