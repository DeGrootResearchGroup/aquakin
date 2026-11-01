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

Two solves share that loop:

- :func:`forward_solve` integrates the state ``y`` alone.
- :func:`staggered_sensitivity_solve` integrates the state together with its
  forward sensitivity ``S = dy/dtheta`` by the **staggered corrector** (Caracotsios
  & Stewart 1985; the CVODES staggered option, Hindmarsh et al. 2005): each
  implicit stage first converges the state, then solves the sensitivity stage --
  which is *linear* in ``S`` -- with the same factorization of
  ``I - h*gamma*J``. See that function for the details.

The per-step Jacobian ``J = df/dy`` is **still computed by autodiff** (forward-mode
``jax.linearize``, materialised by the colored-AD column compression -- the same
exact Jacobian the differentiable path uses, so the step behaviour matches). It
only steers the simplified Newton iterations: every residual is evaluated exactly,
so an approximate ``J`` costs iterations, never accuracy. What is dropped is only
the *adjoint over the whole solve*: the result is **not** differentiable w.r.t.
parameters / initial conditions. So this is an opt-in fast lane for non-AD solves;
any differentiated solve must use the ``diffrax`` path.

The run is faster than the colored ``diffrax`` forward path -- measured ~0.86x
(compile+run) and ~0.88x (run-only) on the full 609-day BSM2 dynamic solve at the
benchmark hourly save grid -- because the lean loop strips the differentiability
scaffolding while the **dense-output continuous extension** (see below) removes
the only remaining per-save penalty. The per-step costs are otherwise the same as
``diffrax`` (both build the colored Jacobian once per step, which XLA fuses into
one batched JVP -- so freezing/reusing it does not help and in fact hurts via
degraded Newton convergence). The compile is also faster, though the margin is
**scale-dependent**: large for small/simple models (where the ``diffrax``
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
2nd-order error estimate (Kvaerno 2004) and an adaptive step controller
(:class:`StepControl`) with two parts: an error controller -- the I-term
``en^(-1/3)`` with a Soderlind PI memory term ``(en_prev/en)^pi_beta`` (the
proportional part damps the overshoot-then-reject oscillation a bare
I-controller suffers; ``pi_beta=0`` recovers the pure I-controller) -- and a
**Gustafsson-style convergence-aware growth limiter** (the simplified-Newton
contraction rate caps the step growth, so a step is rarely grown into
nonlinear-divergence, the chord's failure mode; diffrax's ``PIDController`` has
no such term, and cannot have one: diffrax hands its controller only the error
estimate, never the Newton contraction rate). The default ``maxfac=2.0`` /
``pi_beta=0.08`` were tuned on the full 609-day BSM2 dynamic run, where they cut
the step count ~20% versus the earlier ``maxfac=5`` pure-I controller (fewer
error-test rejections).

The simplified-Newton inner tolerance ``_KAPPA = 1e-1`` was likewise tuned on that
run: loosening it from ``1e-2`` cut the Newton iterations per stage ~3.7 -> ~3.3
(and the step count fell slightly too, so it is a pure win, not a rejection
trade-off), ~10% faster, while the solution stays within ``rtol`` -- the
final-state difference from the ``diffrax`` solve is ~2.5e-4, far inside the
benchmark agreement. The RHS evaluation (recycle + kinetics), at ~12 per step, is
the dominant cost; the per-step Jacobian factorization is only ~8% (so block /
sparse factorization is not worth it at this scale -- it loses to XLA's dense LU).

The stage iteration can be accelerated with the step's factorization as the
preconditioner -- by Anderson mixing (``_ANDERSON_DEPTH``) or by Newton-Krylov
updates solved with GMRES (``_KRYLOV_SWITCH_RATE``), both off by default; see
:func:`_stage_solve`. They target stages where the Jacobian factored at the step
start has gone wrong in a few directions by the stage value -- the plain
iteration then contracts slowly, stalls at the iteration cap or diverges, while
the accelerated one removes those directions in about as many iterations.

References
----------
- Kvaerno, A. (2004). Singly diagonally implicit Runge-Kutta methods with an
  explicit first stage. BIT 44, 489-502.
- Hairer, E. & Wanner, G. (1996). Solving Ordinary Differential Equations II,
  2nd ed., Section IV.8 (simplified Newton convergence test).
- Gustafsson, K. (1994). Control-theoretic techniques for stepsize selection in
  implicit Runge-Kutta methods. ACM TOMS 20(4), 496-517.
- Caracotsios, M. & Stewart, W.E. (1985). Sensitivity analysis of initial value
  problems with mixed ODEs and algebraic equations. Comput. Chem. Eng. 9(4),
  359-365. (The staggered direct sensitivity method.)
- Hindmarsh, A.C. et al. (2005). SUNDIALS: Suite of Nonlinear and
  Differential/Algebraic Equation Solvers. ACM TOMS 31(3), 363-396. (CVODES
  staggered corrector and its sensitivity error norm.)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import equinox as eqx
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
_AL31, _AL32 = 1.0 - _TH, _TH  # stage-3 predictor
_AL41, _AL42, _AL43 = _A31, _A32, _G  # stage-4 predictor
_C2, _C3, _C4 = 2 * _G, 1.0, 1.0  # stage times
_B = (_A41, _A42, _A43, _G)  # solution weights (stiffly accurate)
_BE = (_A41 - _A31, _A42 - _A32, _A43 - _G, _G)  # embedded-error weights
# Row i holds the explicit part a_ij (j < i) of stage i, and the weights of the
# starting guess for its Newton iteration; stage 0 is the explicit first stage.
_A_ROWS = ((), (_A21,), (_A31, _A32), (_A41, _A42, _A43))
_GUESS_ROWS = ((), (1.0,), (_AL31, _AL32), (_AL41, _AL42, _AL43))
_C = (0.0, _C2, _C3, _C4)

_MAXNEWT = 12
_KAPPA = 1e-1  # simplified-Newton convergence tolerance (Hairer eta test)
_ETA_RATE_FLOOR = 1e-3  # floor on (1 - rate) so the Hairer convergence-rate
#                         estimate eta stays finite as the Newton rate -> 1
_SAVE_TIME_RTOL = 1e-12  # relative slack when testing whether a save time falls
#                          within the just-accepted step
_H_STUCK = 1e-13  # step size below which the solve is declared stuck (no progress)
_NONCONVERGED_SHRINK = 0.25  # step factor after a non-converged Newton iteration
_ANDERSON_DEPTH = 0  # Anderson acceleration depth of the stage iteration (0 = off)
_ANDERSON_REG = 1e-10  # relative Tikhonov term on the Anderson least squares
# Newton-Krylov updates of the stage iteration: None = off; a negative value =
# every update; a rate r in [0, 1) = from the first iteration whose contraction
# rate exceeds r on (the earlier updates are plain steps).
_KRYLOV_SWITCH_RATE = None
_KRYLOV_TOL = 1e-2  # GMRES relative residual target of one Newton-Krylov update
_KRYLOV_MAX_ITER = 6  # GMRES iterations per Newton-Krylov update
_GMRES_REG = 1e-14  # relative Tikhonov term on the GMRES least squares


@dataclass(frozen=True)
class StepControl:
    """Step-size control and step budget of the lean ESDIRK loop.

    One value shared by :func:`forward_solve` and
    :func:`staggered_sensitivity_solve`, so the two solves cannot drift in how
    they choose their steps.

    Parameters
    ----------
    h0 : float
        Initial step size.
    theta_target : float
        Target simplified-Newton contraction rate. The convergence-aware limiter
        caps the growth factor at ``theta_target / rate``, so a step whose Newton
        iteration contracted slowly is not grown into divergence.
    safety : float
        Safety factor on the error-controller proposal.
    minfac, maxfac : float
        Bounds on the per-step growth factor.
    pi_beta : float
        Exponent of the Soderlind PI memory term ``(en_prev / en)**pi_beta``;
        ``0`` gives the pure I-controller.
    dtmax : float, optional
        Largest step size. ``None`` leaves the step uncapped.
    max_steps : int
        Step budget, counting accepted and rejected steps.
    """

    h0: float = 1e-3
    theta_target: float = 0.3
    safety: float = 0.9
    minfac: float = 0.2
    maxfac: float = 2.0
    pi_beta: float = 0.08
    dtmax: float | None = None
    max_steps: int = 100_000_000

    def factor(self, en, en_prev, rate, conv):
        """Step-size factor after an attempt with error norm ``en``, previous
        accepted error norm ``en_prev`` and Newton contraction rate ``rate``.

        The error controller is an I-term ``en^(-1/3)`` with the Soderlind PI
        memory term ``(en_prev/en)^pi_beta`` -- the proportional part that damps
        the controller's overshoot-then-reject oscillation a bare I-controller
        suffers. It is combined with the Gustafsson-style convergence-rate
        growth limiter ``theta_target / rate``, which caps growth by the
        simplified-Newton contraction rate so a step is not grown into nonlinear
        divergence. A non-converged attempt (``conv`` false) shrinks the step
        hard.
        """
        ens = jnp.maximum(en, 1e-30)
        f_pi = (en_prev / ens) ** self.pi_beta
        f_err = self.safety * jnp.where(en > 0, ens ** (-1.0 / 3.0) * f_pi, self.maxfac)
        f_conv = jnp.where(rate > 1e-3, self.theta_target / rate, self.maxfac)
        fac = jnp.clip(jnp.minimum(f_err, f_conv), self.minfac, self.maxfac)
        return jnp.where(conv, fac, _NONCONVERGED_SHRINK)


class SolveStats(NamedTuple):
    """Step counts of a lean ESDIRK solve (each a 0-d integer array).

    Attributes
    ----------
    attempted : jax.Array
        Steps attempted (accepted plus rejected).
    accepted : jax.Array
        Steps accepted.
    nonconverged : jax.Array
        Attempts rejected because a simplified Newton iteration did not converge
        (or the candidate state was not finite).
    error_rejected : jax.Array
        Attempts whose Newton iterations converged but which failed the error
        test.
    """

    attempted: jax.Array
    accepted: jax.Array
    nonconverged: jax.Array
    error_rejected: jax.Array


class SensitivitySolve(NamedTuple):
    """Result of :func:`staggered_sensitivity_solve`.

    Attributes
    ----------
    ys : jnp.ndarray
        State at the save times, shape ``(n_t, n)``.
    S : jnp.ndarray
        Sensitivity ``dy/dtheta`` at the save times, shape ``(n_t, n, k)``.
    stats : SolveStats
        Step counts of the solve.
    """

    ys: jnp.ndarray
    S: jnp.ndarray
    stats: SolveStats


class _Attempt(NamedTuple):
    """One attempted step from ``t`` to ``t + h``."""

    z: Any  # candidate solution at t + h (an array or a tuple of arrays)
    en: jax.Array  # error norm; the step is accepted when en <= 1
    rate: jax.Array  # largest simplified-Newton contraction rate in the step
    conv: jax.Array  # every Newton iteration converged and the candidate is finite
    f0: Any  # slope at t (dense output), same structure as z
    f1: Any  # slope at t + h (dense output)


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


def _wrms(v, scale):
    """Weighted root-mean-square norm ``sqrt(mean((v / scale)**2))``."""
    return jnp.sqrt(jnp.mean((v / scale) ** 2))


def _wrms_max_row(v, scale):
    """Largest per-row weighted RMS norm of a ``(k, n)`` block -- the CVODES
    sensitivity norm, so no single parameter's column is averaged away."""
    return jnp.max(jnp.sqrt(jnp.mean((v / scale) ** 2, axis=-1)), initial=0.0)


def _weighted(weights, ks):
    """``sum_j weights[j] * ks[j]`` over the leading stages."""
    return sum(w * k for w, k in zip(weights, ks))


def _anderson_correction(f, dF, dX, valid, weight):
    """The Anderson (Type II) correction for one row of the stage iteration.

    With ``f`` the plain chord step at the current iterate and ``dF`` / ``dX``
    the last differences of steps and iterates (one per row of the history,
    ``valid`` marking the filled ones), returns ``(dX + dF) @ gamma`` with
    ``gamma`` the least-squares fit of ``f`` by ``dF`` in the weighted norm
    (Walker & Ni 2011). The accelerated step is ``f`` minus this.
    """
    A = dF * weight  # (m, n)
    G = A @ A.T
    rhs = A @ (f * weight)
    lam = _ANDERSON_REG * jnp.trace(G) + 1e-300
    G = G * jnp.outer(valid, valid) + jnp.diag(jnp.where(valid, lam, 1.0))
    gamma = jnp.linalg.solve(G, jnp.where(valid, rhs, 0.0))
    return gamma @ (dX + dF)


def _gmres(apply_B, b, weight, tol, max_iter):
    """GMRES for ``B x = b``, the rows of ``b`` solved in lockstep.

    Each row ``b[r]`` (shape ``(n,)``) is its own system with the same
    row-wise operator: ``apply_B`` maps a ``(rows, n)`` block to ``B`` applied to
    every row. Inner products are weighted by ``weight`` (so the residual is
    minimized in the solver's error norm), the Krylov basis is built by modified
    Gram-Schmidt, and the iteration stops once every row's residual is below
    ``tol`` times its right-hand side, or after ``max_iter`` iterations. The
    starting guess is zero, so the first iterate is ``b`` scaled.

    Returns
    -------
    x : jnp.ndarray
        The minimal-residual iterate, shape ``(rows, n)``.
    iterations : jax.Array
        The number of GMRES iterations taken.
    """
    rows, n = b.shape
    m = max_iter

    def wdot(u, v):
        return jnp.sum((weight * u) * (weight * v), axis=-1)

    beta = jnp.sqrt(wdot(b, b))
    V = jnp.zeros((m + 1, rows, n), b.dtype).at[0].set(b / jnp.where(beta > 0, beta, 1.0)[:, None])
    H = jnp.zeros((rows, m + 1, m), b.dtype)
    e1 = jnp.zeros((m + 1,), b.dtype).at[0].set(1.0)

    def least_squares(H, j):
        # min_y ||beta e1 - H y|| over the first j columns (the basis is
        # orthonormal in the weighted inner product, so this is the residual).
        cols = jnp.arange(m) < j
        Hc = H * cols
        G = jnp.einsum("rkm,rkl->rml", Hc, Hc)
        lam = _GMRES_REG * jnp.trace(G, axis1=1, axis2=2) + 1e-300
        G = G * jnp.outer(cols, cols) + jnp.where(cols, lam[:, None], 1.0)[:, :, None] * jnp.eye(m)
        y = jnp.linalg.solve(G, (beta[:, None] * Hc[:, 0, :])[..., None])[..., 0] * cols
        res = beta[:, None] * e1 - jnp.einsum("rkm,rm->rk", Hc, y)
        return y, jnp.sqrt(jnp.sum(res**2, axis=-1))

    def cond(c):
        _, _, j, res = c
        return (j < m) & jnp.any(res > tol * beta)

    def body(c):
        V, H, j, _ = c

        def project(i, wh):
            w, H = wh
            h = jnp.where(i <= j, wdot(w, V[i]), 0.0)
            return w - h[:, None] * V[i], H.at[:, i, j].set(h)

        w, H = jax.lax.fori_loop(0, m, project, (apply_B(V[j]), H))
        h_next = jnp.sqrt(wdot(w, w))
        H = H.at[:, j + 1, j].set(h_next)
        V = V.at[j + 1].set(w / jnp.where(h_next > 0, h_next, 1.0)[:, None])
        return V, H, j + 1, least_squares(H, j + 1)[1]

    V, H, j, _ = jax.lax.while_loop(cond, body, (V, H, 0, beta))
    y, _ = least_squares(H, j)
    return jnp.einsum("rm,mrn->rn", y, V[:m]), j


def _stage_solve(residual, guess, lu, scale):
    """Solve a stage equation ``residual(x) = 0`` with the step's factorization
    ``lu = LU(I - h*G*J)`` (the residual's Jacobian up to the error in ``J``).

    ``x`` is one stage derivative ``(n,)`` or a ``(k, n)`` block of them, one per
    row; each row is solved with the same factorization. ``scale`` (the shape of
    ``x``, or broadcastable to it) weights the norm: the WRMS norm of a row, the
    largest over rows for a block. Each iteration forms the plain step
    ``f = -LU^{-1} residual(x)`` -- the preconditioned residual -- and the
    Hairer-Wanner (ODEs II, IV.8) contraction test reads it: converge when
    ``eta*||f|| < KAPPA``; diverge when the rate is ``>= 1`` or ``f`` is not
    finite. The update applied is one of:

    - **the plain step** ``f`` (the simplified Newton, or chord, iteration);
    - **Anderson-accelerated** (``_ANDERSON_DEPTH > 0``; Walker & Ni 2011, Type
      II, row by row): ``f`` corrected by the least-squares combination of the
      last ``_ANDERSON_DEPTH`` steps that best cancels it;
    - **Newton-Krylov** (``_KRYLOV_SWITCH_RATE`` not ``None``): the exact Newton
      step ``A(x) dx = -residual(x)``, ``A`` the residual's Jacobian at the
      iterate (Jacobian-vector products of one linearization), solved by GMRES
      (:func:`_gmres`) with the factorization as left preconditioner -- so its
      first Krylov vector is ``f`` itself, and it converges in one GMRES
      iteration when the factored ``J`` is still right. A negative switch rate
      makes every update a Newton-Krylov one; a rate ``r`` in ``[0, 1)`` keeps
      plain steps until the contraction rate first exceeds ``r``. A plain step
      that does not contract then switches instead of diverging; divergence is
      judged on Newton-Krylov updates.

    Both accelerations target stages where the factored ``J`` has gone wrong in
    a few directions by the stage value -- the stage value has moved across a
    switch in the right-hand side since the step start: the plain iteration
    contracts slowly along them, while either acceleration removes them in about
    as many iterations as there are such directions. The tests read ``f``
    rather than the applied update because the accelerated updates are not
    monotone; for plain steps the two coincide.

    Returns
    -------
    x : jnp.ndarray
        The last iterate.
    rate : jax.Array
        The last contraction-rate estimate.
    conv : jax.Array
        Whether the iteration converged.
    iterations : jax.Array
        The number of iterations taken.
    """
    depth = _ANDERSON_DEPTH
    switch_rate = _KRYLOV_SWITCH_RATE
    if depth and switch_rate is not None:
        raise ValueError("Anderson acceleration and Newton-Krylov updates are exclusive.")
    rows = guess.reshape((-1, guess.shape[-1]))
    weight = 1.0 / jnp.broadcast_to(scale, guess.shape).reshape(rows.shape)
    history = jnp.zeros((rows.shape[0], max(depth, 1), rows.shape[1]), guess.dtype)

    def precondition(r):
        return -jsla.lu_solve(lu, r.T).T

    def accelerate(f, x, f_prev, x_prev, dF, dX, it):
        if depth == 0:
            return f, dF, dX
        fresh = it >= 1

        def push(buf, new):
            pushed = jnp.concatenate([new.reshape(rows.shape)[:, None], buf[:, :-1]], axis=1)
            return jnp.where(fresh, pushed, buf)

        dF, dX = push(dF, f - f_prev), push(dX, x - x_prev)
        valid = jnp.arange(depth) < jnp.minimum(it, depth)
        correction = jax.vmap(_anderson_correction, in_axes=(0, 0, 0, None, 0))(
            f.reshape(rows.shape), dF, dX, valid, weight
        )
        return f - correction.reshape(f.shape), dF, dX

    def newton_krylov(x, f):
        _, lin = jax.linearize(residual, x)

        def apply_B(V):  # LU^{-1} A(x), row by row
            return -precondition(lin(V.reshape(x.shape))).reshape(rows.shape)

        dx, _ = _gmres(apply_B, f.reshape(rows.shape), weight, _KRYLOV_TOL, _KRYLOV_MAX_ITER)
        return dx.reshape(x.shape)

    def cond(c):
        conv, div, it = c[-2], c[-1], c[5]
        return (~conv) & (~div) & (it < _MAXNEWT)

    def body(c):
        x, f_prev, x_prev, dF, dX, it, dprev, rate, krylov, conv, div = c
        f = precondition(residual(x))
        d = _wrms_max_row(f, scale)
        rate = jnp.where(it >= 1, d / (dprev + 1e-30), rate)
        eta = rate / jnp.maximum(1.0 - rate, _ETA_RATE_FLOOR)
        conv = (it >= 1) & (rate < 1.0) & (eta * d < _KAPPA)
        if switch_rate is None:
            div = ((it >= 1) & (rate >= 1.0)) | (~jnp.isfinite(d))
            dx, dF, dX = accelerate(f, x, f_prev, x_prev, dF, dX, it)
            return x + dx, f, x, dF, dX, it + 1, d, rate, krylov, conv, div
        # ``krylov``: the update that led to this iterate was Newton-Krylov.
        div = ((it >= 1) & (rate >= 1.0) & krylov) | (~jnp.isfinite(d))
        if switch_rate < 0:
            use = jnp.asarray(True)
        else:
            use = krylov | ((it >= 1) & (rate > switch_rate))
        dx = jax.lax.cond(use & ~conv & ~div, lambda: newton_krylov(x, f), lambda: f)
        return x + dx, f, x, dF, dX, it + 1, d, rate, use, conv, div

    init = (guess, guess, guess, history, history, 0, jnp.inf, 0.5, False, False, False)
    x, _, _, _, _, it, _, rate, _, conv, _ = jax.lax.while_loop(cond, body, init)
    return x, rate, conv, it


def _stages(k1, solve_stage):
    """Run Kvaerno3's implicit stages 2-4 after the explicit first stage ``k1``.

    ``solve_stage(i, acc, guess) -> (k_i, rate, conv, iterations)`` solves implicit stage
    ``i`` given ``acc = sum_{j<i} a_ij k_j`` and the starting guess of its Newton
    iteration. Used for the state and the sensitivity stages alike.

    Returns
    -------
    ks : list of jnp.ndarray
        The four stage derivatives.
    rate : jax.Array
        The largest contraction rate over the implicit stages.
    conv : jax.Array
        Whether every implicit stage converged.
    """
    ks = [k1]
    rates, convs = [], []
    for i in range(1, 4):
        k, rate, conv, _ = solve_stage(i, _weighted(_A_ROWS[i], ks), _weighted(_GUESS_ROWS[i], ks))
        ks.append(k)
        rates.append(rate)
        convs.append(conv)
    rate = jnp.maximum(jnp.maximum(rates[0], rates[1]), rates[2])
    return ks, rate, convs[0] & convs[1] & convs[2]


def _advance(z, ks, h):
    """The step's solution ``z + h*sum(b_i k_i)`` and its embedded error estimate."""
    return z + h * _weighted(_B, ks), h * _weighted(_BE, ks)


def _state_stages(rhs, args, t, y, h, lu, scale):
    """The state stage derivatives of one Kvaerno3 step from ``(t, y)``.

    Returns the four stage derivatives, the four stage values ``Y_i`` (``Y_0 =
    y``; the stiffly-accurate ``Y_3`` is the step's solution), the largest
    contraction rate, and whether every stage converged.
    """
    k1 = rhs(t, y, args)

    def solve_stage(i, acc, guess):
        t_i = t + _C[i] * h
        base = y + h * acc
        return _stage_solve(lambda k: k - rhs(t_i, base + h * _G * k, args), guess, lu, scale)

    ks, rate, conv = _stages(k1, solve_stage)
    Ys = [y] + [y + h * (_weighted(_A_ROWS[i], ks) + _G * ks[i]) for i in range(1, 4)]
    return ks, Ys, rate, conv


def _factor(jac, t, y, args, h):
    """``LU(I - h*G*J)`` with ``J = jac(t, y, args)``, the step's Newton operator."""
    J = jac(t, y, args)
    return jsla.lu_factor(jnp.eye(J.shape[0], dtype=J.dtype) - h * _G * J)


def _state_attempt(rhs, jac, args, rtol, atol):
    """Step attempt for the state alone (:func:`forward_solve`)."""

    def attempt(t, y, h):
        lu = _factor(jac, t, y, args, h)
        scale = atol + rtol * jnp.abs(y)
        ks, _, rate, conv = _state_stages(rhs, args, t, y, h, lu, scale)
        y1, err = _advance(y, ks, h)
        conv = conv & jnp.all(jnp.isfinite(y1))
        # k1 (slope at t) and k4 (slope at t+h, the stiffly-accurate endpoint
        # stage) feed the cubic-Hermite dense output.
        return _Attempt(y1, _wrms(err, scale), rate, conv, ks[0], ks[3])

    return attempt


def _staggered_attempt(rhs, jac, theta, tangents, rtol, atol, sens_rtol, sens_atol):
    """Step attempt for the state plus its sensitivity, staggered corrector.

    The state stages converge first. Only if they converged and the state passes
    its error test are the sensitivity stages solved; otherwise the attempt is
    rejected on the state alone, without spending the sensitivity work on it.
    """

    def tangent_map(t_i, Y):
        # Linearize f once at the stage value; each sensitivity column is the
        # same linear map applied to (S_j, e_j): J(Y) S_j + f_theta(Y)_j.
        _, lin = jax.linearize(lambda yy, pp: rhs(t_i, yy, pp), Y, theta)
        return lambda V: jax.vmap(lin)(V, tangents)

    def attempt(t, z, h):
        y, S = z
        lu = _factor(jac, t, y, theta, h)
        scale_y = atol + rtol * jnp.abs(y)
        ks, Ys, rate_y, conv_y = _state_stages(rhs, theta, t, y, h, lu, scale_y)
        y1, err_y = _advance(y, ks, h)
        conv_y = conv_y & jnp.all(jnp.isfinite(y1))
        en_y = _wrms(err_y, scale_y)

        def sensitivity(_):
            scale_S = sens_atol + sens_rtol * jnp.abs(S)

            def solve_stage(i, acc, guess):
                # The sensitivity stage is linear in K: K = J(Y_i)(base + h*G*K)
                # + f_theta(Y_i). The chord iteration reuses the state's factor;
                # it converges in one correction when J(Y_i) equals the J that
                # was factored, and in a few when it does not.
                F = tangent_map(t + _C[i] * h, Ys[i])
                base = S + h * acc
                return _stage_solve(lambda K: K - F(base + h * _G * K), guess, lu, scale_S)

            KS, rate_S, conv_S = _stages(tangent_map(t, y)(S), solve_stage)
            S1, err_S = _advance(S, KS, h)
            conv_S = conv_S & jnp.all(jnp.isfinite(S1))
            return S1, _wrms_max_row(err_S, scale_S), rate_S, conv_S, KS[0], KS[3]

        def skipped(_):
            zeros = jnp.zeros_like(S)
            return S, jnp.zeros_like(en_y), jnp.zeros_like(rate_y), jnp.asarray(True), zeros, zeros

        S1, en_S, rate_S, conv_S, KS1, KS4 = jax.lax.cond(
            conv_y & (en_y <= 1.0), sensitivity, skipped, None
        )
        return _Attempt(
            (y1, S1),
            jnp.maximum(en_y, en_S),
            jnp.maximum(rate_y, rate_S),
            conv_y & conv_S,
            (ks[0], KS1),
            (ks[3], KS4),
        )

    return attempt


def _integrate(attempt, z0, t0, t1, t_eval, control: StepControl):
    """The adaptive loop: step-size control, dense output and save recording.

    ``attempt(t, z, h) -> _Attempt`` takes one step; ``z`` is an array or a tuple
    of arrays, each recorded at the save times. Returns the saved values (each
    leaf with a leading ``len(t_eval)`` axis) and the :class:`SolveStats`.
    """
    t0 = jnp.asarray(t0, dtype=float)
    t1 = jnp.asarray(t1, dtype=float)
    t_eval = jnp.asarray(t_eval)
    n_save = t_eval.shape[0]
    zs0 = jax.tree.map(lambda a: jnp.zeros((n_save,) + a.shape, a.dtype), z0)
    max_steps = control.max_steps

    def cap(h):
        return h if control.dtmax is None else jnp.minimum(h, control.dtmax)

    def record(zs, i, z):
        return jax.tree.map(lambda buf, a: buf.at[i].set(a), zs, z)

    def _tol(s):
        return _SAVE_TIME_RTOL * jnp.maximum(1.0, jnp.abs(s))

    # carry: t, z, h_ctrl (unclipped proposal), save_idx, zs, en_prev (last
    #        accepted error, for the PI term), dead, stats
    def cond(c):
        t, z, h_ctrl, sidx, zs, en_prev, dead, stats = c
        return (sidx < n_save) & (~dead) & (stats.attempted < max_steps)

    def body(c):
        t, z, h_ctrl, sidx, zs, en_prev, dead, stats = c
        next_save = t_eval[sidx]

        def record_only():
            # already at this save time (e.g. t_eval[0] == t0): record, advance,
            # do not take a (zero-width) step.
            return (t, z, h_ctrl, sidx + 1, record(zs, sidx, z), en_prev, dead, stats)

        def do_step():
            # Take the NATURAL adaptive step, clipped only to the final time t1
            # (not to each save). Save points that fall inside the step are
            # recovered by the cubic-Hermite continuous extension below, so the
            # controller keeps its natural step rhythm instead of being forced to
            # plant an extra boundary on every t_eval point (the step-clipping
            # cost at a dense save grid -- issue #386).
            h = jnp.minimum(h_ctrl, t1 - t)
            a = attempt(t, z, h)
            accept = a.conv & (a.en <= 1.0)
            h_ctrl_new = cap(h_ctrl * control.factor(a.en, en_prev, a.rate, a.conv))
            t_new = jnp.where(accept, t + h, t)
            z_new = jax.tree.map(lambda new, old: jnp.where(accept, new, old), a.z, z)
            # The PI memory is updated only on accept.
            en_prev_new = jnp.where(accept, jnp.maximum(a.en, 1e-30), en_prev)

            # Dense output: on an accepted step, record EVERY save time in
            # (t, t_new] by interpolating within the step (a step may span several
            # saves at a sparse grid, or none at a dense one).
            def rec_cond(rc):
                zs_, si_ = rc
                in_range = si_ < n_save
                tsi = t_eval[jnp.minimum(si_, n_save - 1)]
                return accept & in_range & (tsi <= t_new + _tol(t_new))

            def rec_body(rc):
                zs_, si_ = rc
                theta = (t_eval[si_] - t) / h
                zsave = jax.tree.map(
                    lambda a0, a1, s0, s1: _hermite(a0, a1, s0, s1, h, theta),
                    z,
                    z_new,
                    a.f0,
                    a.f1,
                )
                return record(zs_, si_, zsave), si_ + 1

            zs_new, sidx_new = jax.lax.while_loop(rec_cond, rec_body, (zs, sidx))
            stuck = h_ctrl_new < _H_STUCK
            stats_new = SolveStats(
                stats.attempted + 1,
                stats.accepted + accept.astype(int),
                stats.nonconverged + (~a.conv).astype(int),
                stats.error_rejected + (a.conv & (a.en > 1.0)).astype(int),
            )
            return (
                t_new,
                z_new,
                h_ctrl_new,
                sidx_new,
                zs_new,
                en_prev_new,
                dead | stuck,
                stats_new,
            )

        already = (next_save - t) <= _tol(next_save)
        return jax.lax.cond(already, record_only, do_step)

    zero = jnp.array(0)
    init = (
        t0,
        z0,
        cap(jnp.asarray(float(control.h0))),
        zero,
        zs0,
        jnp.array(1.0),
        jnp.array(False),
        SolveStats(zero, zero, zero, zero),
    )
    t, z, h, sidx, zs, en_prev, dead, stats = jax.lax.while_loop(cond, body, init)
    # A loop that exits before filling every save row (step budget used up, or
    # the step size collapsed) must not hand back the zero-initialized rows as if
    # they were a solution: mark them NaN and raise.
    incomplete = sidx < n_save
    filled = jnp.arange(n_save) < sidx

    def mark(buf):
        mask = filled.reshape((n_save,) + (1,) * (buf.ndim - 1))
        return jnp.where(mask, buf, jnp.nan)

    zs = jax.tree.map(mark, zs)
    zs = eqx.error_if(
        zs,
        incomplete & (stats.attempted >= max_steps),
        f"forward_solve: the maximum number of solver steps was reached "
        f"(max_steps={max_steps}) before the last save time.",
    )
    zs = eqx.error_if(
        zs,
        incomplete,
        "forward_solve: the solve stalled before the last save time (the step "
        "size collapsed because the Newton iteration kept failing or the state "
        "became non-finite).",
    )
    return zs, stats


def forward_solve(
    rhs: Callable,
    jac: Callable,
    y0: jnp.ndarray,
    args,
    t0,
    t1,
    t_eval: jnp.ndarray,
    *,
    rtol,
    atol,
    control: StepControl = StepControl(),
):
    """Integrate ``dy/dt = rhs(t, y, args)`` from ``t0`` to ``t1``, forward only.

    Parameters
    ----------
    rhs : Callable
        ``rhs(t, y, args) -> dy/dt``.
    jac : Callable
        ``jac(t, y, args) -> J`` the dense ``(n, n)`` Jacobian ``df/dy`` (formed
        by colored forward AD by the caller). It only steers the simplified
        Newton iteration, so an approximation costs iterations, not accuracy.
    y0 : jnp.ndarray
        Initial state ``(n,)``.
    args :
        Threaded to ``rhs`` / ``jac`` (the parameter vector) -- a runtime
        argument, so one compile serves a parameter sweep.
    t0, t1 : float or jax.Array
        Integration interval (may be traced).
    t_eval : jnp.ndarray
        Strictly increasing save times in ``[t0, t1]``; the solution is returned
        at exactly these times (by dense output within each step).
    rtol, atol : float or array
        Error tolerances (WRMS norm ``atol + rtol*|y|``).
    control : StepControl, optional
        Step-size controller settings and step budget.

    Returns
    -------
    ys : jnp.ndarray
        Solution at ``t_eval``, shape ``(len(t_eval), n)``.

    Raises
    ------
    equinox.EquinoxRuntimeError
        If the solve stops before reaching the last save time, either because it
        used up ``control.max_steps`` (the message contains "maximum number of
        solver steps", so
        :func:`~aquakin.integrate._common.friendly_solve_errors` re-raises it
        with remedies) or because the step size collapsed (the Newton iteration
        kept failing or the state became non-finite). The check runs inside the
        compiled solve, so it also works under ``jax.jit``.
    """
    atol = jnp.asarray(atol)
    ys, _ = _integrate(_state_attempt(rhs, jac, args, rtol, atol), y0, t0, t1, t_eval, control)
    return ys


def staggered_sensitivity_solve(
    rhs: Callable,
    jac: Callable,
    y0: jnp.ndarray,
    theta: jnp.ndarray,
    free_idx: jnp.ndarray,
    t0,
    t1,
    t_eval: jnp.ndarray,
    *,
    rtol,
    atol,
    sens_rtol=None,
    sens_atol=None,
    param_scale=None,
    control: StepControl = StepControl(),
) -> SensitivitySolve:
    """Integrate ``y`` and its forward sensitivity ``S = dy/dtheta`` (staggered).

    ``S`` obeys the variational equation ``dS/dt = J(t, y) S + f_theta(t, y)``.
    Applying the Kvaerno3 ESDIRK to the augmented system ``[y; S]`` gives, for
    each implicit stage ``i`` with state stage value ``Y_i``, a sensitivity stage
    that is *linear* in its stage derivative ``K``::

        K = J(Y_i) (S_n + h * sum_{j<i} a_ij K_j + h*gamma*K) + f_theta(Y_i)

    The **staggered corrector** solves the state stage first (simplified Newton
    with ``LU(I - h*gamma*J)``, ``J`` at the step start), then this linear stage
    with the **same** factorization, column by column. The residual is evaluated
    exactly -- one linearization of ``f`` at ``(Y_i, theta)`` per stage, pushed
    along the ``k`` parameter columns -- so the result is the augmented-system
    stage to the Newton tolerance, while ``J`` only steers the iterations. No
    second derivatives of ``f`` are formed, so the per-step cost grows with
    ``k`` only through the ``k`` Jacobian-vector products per iteration, never
    through a factorization of the ``n(1+k)`` augmented operator or a probe of
    its coupling blocks.

    The error test bounds both blocks, with the CVODES norm: the step is
    accepted when the state's WRMS error and every sensitivity column's WRMS
    error are below one. The state's error test runs first; a step rejected on
    the state alone is rejected before any sensitivity work is spent on it. The
    step controller is the one :func:`forward_solve` uses, including the
    convergence-aware growth limiter, whose contraction rate covers the state
    and sensitivity iterations.

    Parameters
    ----------
    rhs : Callable
        ``rhs(t, y, theta) -> dy/dt``, depending on ``theta`` wherever the
        parameters enter (so its Jacobian-vector product carries the full
        sensitivity).
    jac : Callable
        ``jac(t, y, theta) -> J`` the ``(n, n)`` Jacobian ``df/dy`` (or an
        approximation; it only steers the iterations).
    y0 : jnp.ndarray
        Initial state ``(n,)``. The initial sensitivity is zero.
    theta : jnp.ndarray
        Parameter vector, shape ``(n_params,)``.
    free_idx : jnp.ndarray
        Integer indices (shape ``(k,)``) of the sensitivity parameters in
        ``theta``.
    t0, t1 : float or jax.Array
        Integration interval (may be traced).
    t_eval : jnp.ndarray
        Strictly increasing save times in ``[t0, t1]``.
    rtol, atol : float or array
        State tolerances (WRMS norm ``atol + rtol*|y|``).
    sens_rtol : float, optional
        Relative tolerance on ``S``; defaults to ``rtol``.
    sens_atol, param_scale : optional
        Absolute tolerance on ``S``; see
        :func:`~aquakin.integrate.forward_sensitivity.sensitivity_atol`
        (default ``atol / |theta_j|`` per column).
    control : StepControl, optional
        Step-size controller settings and step budget.

    Returns
    -------
    SensitivitySolve
        ``ys`` ``(n_t, n)``, ``S`` ``(n_t, n, k)`` and the step counts.

    Raises
    ------
    equinox.EquinoxRuntimeError
        As :func:`forward_solve`.
    """
    from aquakin.integrate.forward_sensitivity import sensitivity_atol

    n = int(y0.shape[0])
    k = int(free_idx.shape[0])
    theta = jnp.asarray(theta)
    atol = jnp.broadcast_to(jnp.asarray(atol, dtype=float), (n,))
    if k == 0:
        ys, stats = _integrate(
            _state_attempt(rhs, jac, theta, rtol, atol), y0, t0, t1, t_eval, control
        )
        return SensitivitySolve(ys, jnp.zeros(ys.shape + (0,), ys.dtype), stats)
    # Unit parameter tangents: row j selects free parameter free_idx[j].
    tangents = jnp.zeros((k, theta.shape[0]), theta.dtype).at[jnp.arange(k), free_idx].set(1.0)
    atol_S = sensitivity_atol(atol, theta[free_idx], sens_atol=sens_atol, param_scale=param_scale)
    attempt = _staggered_attempt(
        rhs,
        jac,
        theta,
        tangents,
        rtol,
        atol,
        rtol if sens_rtol is None else sens_rtol,
        atol_S,
    )
    S0 = jnp.zeros((k, n), y0.dtype)  # row j = sensitivity column j
    (ys, Ss), stats = _integrate(attempt, (y0, S0), t0, t1, t_eval, control)
    return SensitivitySolve(ys, jnp.swapaxes(Ss, 1, 2), stats)
