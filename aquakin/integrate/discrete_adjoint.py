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
``Kvaerno5``, the method the reactors use). The forward saves each step's stage
derivatives (diffrax dense output), so the discrete adjoint reconstructs the
stage values exactly in the backward pass -- no Newton recompute -- and applies
the transposed-stage recurrence (see that function). Either way every per-step
solve is the same well-conditioned
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

from aquakin.integrate._common import validate_t_eval

# Shared forward-solve defaults for both discrete-adjoint solvers.
_DEFAULT_RTOL = 1e-6              # PID controller relative tolerance
_DEFAULT_ATOL = 1e-9             # PID controller absolute tolerance
_DEFAULT_DT0 = 1e-6             # initial step (the adaptive controller grows it)
_DEFAULT_MAX_STEPS = 200_000   # saved-trajectory buffer the backward scan walks


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


def _discrete_adjoint_solve(
    rhs,
    y0,
    params,
    t_span,
    t_eval,
    *,
    solver,
    step_adjoint,
    rtol,
    atol,
    dt0,
    max_steps,
    save_stages=False,
):
    """Shared forward + discrete-adjoint backward harness for both solvers.

    The forward pass is a robust adaptive diffrax solve with ``solver``, forced
    to land steps exactly on the observation times (so each is a step boundary
    and the adjoint needs no interpolation). The custom VJP is the exact
    discrete adjoint, evaluated by a backward scan of bounded transposed solves
    -- finite for stiff networks at any step size, with no ``dtmax`` cap.

    Everything here is method-independent; the only per-method piece is
    ``step_adjoint(...) -> (lam_n, dpar)``, the single-step transposed-solve body
    (implicit Euler inlines one solve on the post-step state ``y_k``; ESDIRK
    reconstructs its stages from the pre-step state and the saved stage
    derivatives). With ``save_stages=True`` the forward stores each step's
    dense-output stage derivatives ``k`` and the driver passes them as a trailing
    argument to ``step_adjoint`` (so the ESDIRK body needs no Newton recompute).
    It is invoked under :func:`jax.lax.cond` so padded /
    invalid trajectory slots are skipped and the backward cost tracks the real
    step count, not the allocated ``max_steps`` buffer.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    n = y0.shape[0]
    final_only = t_eval is None
    if not final_only:
        # Out-of-span / non-ascending save times otherwise silently return inf /
        # wrong values (the backward scan injects cotangents only at landed
        # steps), poisoning any downstream loss. Validate as the reactors do.
        validate_t_eval(jnp.asarray(t_eval), t0, t1)
    teval = jnp.asarray([t1] if final_only else t_eval, dtype=jnp.result_type(float))

    term = diffrax.ODETerm(lambda t, y, a: rhs(t, y, a))
    controller = diffrax.ClipStepSizeController(
        diffrax.PIDController(rtol=rtol, atol=atol), step_ts=teval
    )

    # ``dense=save_stages`` makes diffrax also store each step's dense-output
    # info, which for a Runge-Kutta solver carries the stage derivatives ``k`` --
    # so an ESDIRK backward can reconstruct its stage values exactly instead of
    # re-solving them by Newton (the dominant backward cost).
    saveat = diffrax.SaveAt(steps=True, dense=save_stages)

    def _forward(y0_, params_):
        return diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0_, args=params_,
            stepsize_controller=controller,
            saveat=saveat, max_steps=max_steps,
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
        if save_stages:
            # Per-step stage derivatives k_j = f(Y_j), shape (max_steps, s, n).
            ks = sol.interpolation.infos["k"]
            return ys_eval, (sol.ts, sol.ys, y0_, params_, idx, ks)
        return ys_eval, (sol.ts, sol.ys, y0_, params_, idx)

    def solve_bwd(res, ybar):
        if save_stages:
            ts, ys, y0_, params_, idx, ks = res  # ybar: (n_obs, n)
        else:
            ts, ys, y0_, params_, idx = res
        K = ts.shape[0]
        valid = jnp.isfinite(ts)
        t_prev = jnp.concatenate([jnp.array([t0], dtype=ts.dtype), ts[:-1]])
        dts = ts - t_prev
        y_prev = jnp.concatenate([y0_[None, :], ys[:-1]], axis=0)

        # Distribute observation cotangents: idx==0 -> y0 directly; idx==m>=1 ->
        # the state produced by step m-1.
        injected = jnp.zeros((K, n), dtype=ys.dtype)
        ybar0_obs = jnp.zeros((n,), dtype=ys.dtype)
        for i in range(teval.shape[0]):
            step_idx = jnp.maximum(idx[i] - 1, 0)
            injected = injected.at[step_idx].add(
                jnp.where(idx[i] >= 1, ybar[i], 0.0)
            )
            ybar0_obs = ybar0_obs + jnp.where(idx[i] == 0, ybar[i], 0.0)

        def body(lam_pbar, k):
            lam, pbar = lam_pbar
            ok = valid[k] & (dts[k] > 0)
            lam_k = lam + injected[k]            # add this step's observation cotangent

            def do(_):
                if save_stages:
                    lam_n, dpar = step_adjoint(
                        y_prev[k], ys[k], params_, dts[k], lam_k, ks[k])
                else:
                    lam_n, dpar = step_adjoint(
                        y_prev[k], ys[k], params_, dts[k], lam_k)
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


def _autonomize(rhs: Callable, y0: jnp.ndarray, t0: float):
    """Carry time in the state so a non-autonomous ``rhs(t, y, p)`` becomes an
    autonomous field on the augmented state ``[y; tau]`` with ``dtau/dt = 1``.

    The wrapped field reads the integration time from ``tau`` (the last state
    component) and ignores the solver's ``t`` argument, so the discrete adjoint --
    which evaluates the field with a fixed time argument -- still captures the
    exact ``df/dt`` dependence through ``tau``. This is the classical
    autonomization that makes the discrete-adjoint construction exact for a
    time-dependent right-hand side, with no change to the per-step recurrence.
    Returns ``(rhs_aug, y0_aug)``.
    """
    n = y0.shape[0]
    y0_aug = jnp.concatenate([y0, jnp.asarray([t0], dtype=y0.dtype)])

    def rhs_aug(t, y_aug, p):
        dy = rhs(y_aug[n], y_aug[:n], p)
        return jnp.concatenate([dy, jnp.ones((1,), dtype=y_aug.dtype)])

    return rhs_aug, y0_aug


def _augment_atol(atol):
    """Extend an absolute tolerance to the time-augmented state. A scalar tol
    broadcasts unchanged; an array tol gains one entry for the time component
    (reusing the last, since the linear ``tau`` is integrated exactly so its
    tolerance does not constrain the step)."""
    a = jnp.asarray(atol)
    if a.ndim == 0:
        return atol
    return jnp.concatenate([a, a[-1:]])


def implicit_euler_adjoint_solve(
    rhs: Callable,
    y0: jnp.ndarray,
    params: jnp.ndarray,
    t_span: tuple[float, float],
    t_eval: Optional[jnp.ndarray] = None,
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    dt0: float = _DEFAULT_DT0,
    max_steps: int = _DEFAULT_MAX_STEPS,
    time_dependent: bool = False,
    primal_rhs: Optional[Callable] = None,
) -> jnp.ndarray:
    """Integrate over ``t_span`` with a cap-free discrete-adjoint reverse-mode rule.

    The forward pass is a robust adaptive implicit-Euler diffrax solve. The
    custom VJP is the exact discrete adjoint of that solve, evaluated by a
    backward scan of bounded transposed linear solves -- finite for stiff
    networks at any step size, with no ``dtmax`` cap. The first-order ``s=1``
    special case of :func:`esdirk_adjoint_solve`; both share the
    :func:`_discrete_adjoint_solve` harness.

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
    time_dependent : bool, optional
        If ``False`` (default) the right-hand side is assumed autonomous, as the
        reaction RHS is for fixed conditions. If ``True`` the field's explicit
        time dependence (e.g. a time-varying influent) is handled exactly by
        carrying time in the state, so the gradient is exact through a transient
        solve. See :func:`_autonomize`.
    primal_rhs : callable, optional
        Fast alternate RHS for the forward solve and the ``df/dy`` Jacobian, while
        ``rhs`` supplies the ``df/dtheta`` vjp. See :func:`esdirk_adjoint_solve`
        for the full contract (it must match ``rhs`` in value and ``df/dy``; only
        the parameter derivative is taken from ``rhs``; ``stop_gradient`` any
        ``params``-derived value closed over). ``None`` uses ``rhs`` throughout.

    Returns
    -------
    jnp.ndarray
        If ``t_eval is None``, the final state ``y(t1)``, shape ``(n,)``.
        Otherwise the states at ``t_eval``, shape ``(len(t_eval), n)``. Either
        way the result carries the discrete-adjoint VJP w.r.t. ``y0`` and
        ``params``.
    """
    n0 = y0.shape[0]
    if time_dependent:
        t0f = float(t_span[0])
        if primal_rhs is not None:
            primal_rhs = _autonomize(primal_rhs, y0, t0f)[0]
        rhs, y0 = _autonomize(rhs, y0, t0f)
        atol = _augment_atol(atol)
    primal = primal_rhs if primal_rhs is not None else rhs
    n = y0.shape[0]
    solver = diffrax.ImplicitEuler(root_finder=_implicit_tols(rtol, atol))

    def step_adjoint(y_prev_k, y_k, params_, dt, lam_k):
        # Implicit Euler step y_{n+1} = y_n + dt f(y_{n+1}); its adjoint uses the
        # post-step state y_k. mu = (I - dt J^T)^{-1} lam_k, then the parameter
        # cotangent is dt (df/dtheta)^T mu. df/dy from ``primal`` (cached); the
        # df/dtheta vjp from ``rhs`` (recomputes any params-derived sub-term).
        Jf = jax.jacfwd(lambda y: primal(0.0, y, params_))(y_k)
        M = jnp.eye(n, dtype=y_k.dtype) - dt * Jf
        mu = jnp.linalg.solve(M.T, lam_k)
        _, vjp = jax.vjp(lambda q: rhs(0.0, y_k, q), params_)
        dpar = dt * vjp(mu)[0]
        return mu, dpar

    out = _discrete_adjoint_solve(
        primal, y0, params, t_span, t_eval,
        solver=solver, step_adjoint=step_adjoint,
        rtol=rtol, atol=atol, dt0=dt0, max_steps=max_steps,
    )
    return out[..., :n0] if time_dependent else out


# --- High-order ESDIRK discrete adjoint --------------------------------------
#
# The implicit-Euler adjoint above is first order. For accuracy parity with the
# reactors -- which integrate with the high-order ESDIRK ``Kvaerno5`` -- the same
# idea (robust diffrax forward + hand-written discrete adjoint) extends to a
# general s-stage ESDIRK. The forward saves each step's stage derivatives (diffrax
# dense output), so the backward reconstructs the stage values exactly by the
# Butcher linear combination and applies the transposed-stage recurrence -- no
# Newton recompute.
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
    rtol: float = _DEFAULT_RTOL,
    atol: float = _DEFAULT_ATOL,
    dt0: float = _DEFAULT_DT0,
    max_steps: int = _DEFAULT_MAX_STEPS,
    time_dependent: bool = False,
    primal_rhs: Optional[Callable] = None,
    jacobian_builder: Optional[Callable] = None,
) -> jnp.ndarray:
    """Cap-free reverse-mode gradient through a high-order ESDIRK solve.

    Like :func:`implicit_euler_adjoint_solve` but the forward uses a high-order
    ESDIRK method (default :class:`diffrax.Kvaerno5`, matching the reactors), and
    the backward is the transposed-stage discrete adjoint of that method. The
    forward saves each step's stage derivatives via diffrax dense output, so the
    backward reconstructs the stage values exactly by the Butcher linear
    combination ``Y_i = y_n + sum_j A[i,j]*k_j`` (the dense-output ``k`` is the
    dt-scaled stage increment) -- no Newton recompute, which
    was the dominant backward cost -- then the per-stage transposed solves
    accumulate the gradient. Finite for stiff networks with no ``dtmax`` cap.

    Parameters
    ----------
    rhs, y0, params, t_span, t_eval, rtol, atol, dt0, max_steps
        As for :func:`implicit_euler_adjoint_solve`.
    solver : diffrax.AbstractSolver, optional
        The ESDIRK forward solver; must expose a Butcher ``tableau``. Defaults to
        :class:`diffrax.Kvaerno5`.
    time_dependent : bool, optional
        If ``False`` (default) the right-hand side is taken to be autonomous (the
        reaction RHS is, for fixed conditions), so the backward pass evaluates it
        with a fixed time argument and the stage times do not enter. If ``True``
        the field's explicit time dependence (e.g. a time-varying influent) is
        handled exactly by carrying time in the state (:func:`_autonomize`), so
        the gradient is exact through a transient solve.
    primal_rhs : callable, optional
        An alternate right-hand side used for the **forward solve and the
        ``df/dy`` stage Jacobians** -- everything except
        the ``df/dtheta`` parameter vjp, which always uses ``rhs``. It must
        produce the *same values and the same ``df/dy``* as ``rhs`` (so the
        trajectory and the state-cotangent recurrence are unchanged); only the
        *parameter* derivative is taken from ``rhs``. The use case is a RHS with a
        state-invariant but parameter-dependent sub-computation that is expensive
        to repeat -- e.g. a plant's recycle map ``M(params)`` -- which can be
        evaluated **once** and reused for every Jacobian/stage call here, while
        ``rhs`` (which recomputes it) still supplies the exact ``dM/dtheta`` in the
        one place it is needed (the parameter vjp). Because the discrete adjoint
        takes its *entire* parameter gradient from that vjp and uses the stages /
        Jacobians only to propagate the *state* cotangent, the result is the exact
        gradient -- bit-identical to using ``rhs`` everywhere when the cached
        sub-computation equals the recomputed one. The caller MUST
        :func:`jax.lax.stop_gradient` any ``params``-derived value it closes over
        in ``primal_rhs`` (otherwise the closed-over tracer escapes the custom
        VJP). ``None`` (default) uses ``rhs`` for everything (the historic path).
    jacobian_builder : callable, optional
        Builder ``(f, y) -> J`` for the per-stage ``df/dy`` Jacobian used in the
        backward pass (built once per reconstructed stage, for the transposed-
        stage solves), where ``f`` is the (autonomized) ``primal`` right-hand at
        fixed parameters. ``None`` (default) builds the **dense** Jacobian with
        ``jax.jacfwd`` (the historic path, bit-identical). A sparsity-**colored**
        builder -- one Jacobian-vector product per color instead of one per state
        -- cuts the dominant backward cost for a large, block-sparse plant: the
        backward is ~80% Jacobian builds. The builder MUST return a ``J`` equal to
        the dense Jacobian (the colored construction is exact when its sparsity
        pattern is a superset of the true nonzeros; the caller guards this at the
        start state and falls back to dense on a mismatch), so the discrete
        adjoint -- and the gradient -- is unchanged. Affects only the backward
        Jacobian build; the forward solve and the parameter vjp are untouched.

    Returns
    -------
    jnp.ndarray
        Final state ``(n,)`` if ``t_eval is None``, else states at ``t_eval``
        ``(len(t_eval), n)``; carries the discrete-adjoint VJP w.r.t. ``y0`` and
        ``params``.
    """
    n0 = y0.shape[0]
    if time_dependent:
        t0f = float(t_span[0])
        if primal_rhs is not None:
            primal_rhs = _autonomize(primal_rhs, y0, t0f)[0]
        rhs, y0 = _autonomize(rhs, y0, t0f)
        atol = _augment_atol(atol)
    # ``primal`` drives the forward solve + the df/dy stage work (it may cache a
    # state-invariant sub-computation); ``rhs`` always supplies the df/dtheta vjp.
    primal = primal_rhs if primal_rhs is not None else rhs
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
    n = y0.shape[0]

    # The df/dy stage Jacobian builder: dense ``jacfwd`` by default, or the
    # caller's sparsity-colored builder ``(f, y) -> J``. ``jacobian_builder is
    # None`` is a trace-time Python branch, so the dense path is bit-identical to
    # the historic code. ``J`` must equal the dense Jacobian (the colored builder
    # carries a superset sparsity pattern, guarded by the caller).
    def _build_jac(f, y):
        if jacobian_builder is None:
            return jax.jacfwd(f)(y)
        return jacobian_builder(f, y)

    def step_adjoint(y_prev_k, y_k, params_, dt, lam, ks):
        # ESDIRK adjoint: the stage values are RECONSTRUCTED from the saved stage
        # increments ks[j] (diffrax dense-output ``k``, which is the dt-SCALED
        # stage derivative dt*f(Y_j)), then the backward sweep applies the
        # per-stage transposed solves. Each stage satisfies Y_i = y_n +
        # sum_j A[i,j]*k_j with A the full lower-triangular Butcher matrix (the
        # dt is already folded into k), so the reconstruction is exact -- no
        # Newton recompute, the dominant backward cost removed. The post-step
        # state y_k is unused (the stages carry the dependence).
        Ys = y_prev_k[None, :] + (A @ ks)             # (s, n)
        # df/dy stage Jacobians from ``primal`` (cached sub-computation); the
        # df/dtheta vjp below from ``rhs`` (recomputes it -> exact dM/dtheta).
        f = lambda y: primal(0.0, y, params_)
        Js = jax.vmap(lambda Y: _build_jac(f, Y))(Ys)
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

    out = _discrete_adjoint_solve(
        primal, y0, params, t_span, t_eval,
        solver=solver, step_adjoint=step_adjoint,
        rtol=rtol, atol=atol, dt0=dt0, max_steps=max_steps,
        save_stages=True,
    )
    return out[..., :n0] if time_dependent else out
