"""Forward (variational) sensitivity solve: integrate ``S = dy/dtheta`` with ``y``.

Computing ``d(output)/d(params)`` through a stiff reactor solve by ordinary AD
breaks down: above an integrator-step threshold the differentiated solve returns
non-finite values (in both forward and reverse mode), and the workaround is a
global ``dtmax`` cap that forces tiny steps over the *whole* solve even though
the tight step is only needed in small, local stiff regions.

Classical forward sensitivity analysis removes the cap. The sensitivity
``S = dy/dtheta`` obeys the variational equation

    dS/dt = J(y, theta, t) . S + f_theta(y, theta, t),
        J = df/dy  (n x n),   f_theta = df/dtheta  (n x p)

so it can be integrated *alongside* the state ``y`` as one augmented system
``z = [y; vec(S)]``. The adaptive step controller is then made to bound the
error of ``S`` as well as ``y``, so it tightens the step only where the
sensitivity is stiff and runs free elsewhere -- no global cap. Column ``k`` of
the sensitivity RHS is exactly a JVP of ``f`` evaluated at the shared
linearisation point ``(y, theta)``::

    (dS/dt)[:, k] = J . S[:, k] + f_theta[:, k]
                  = jvp(f, (y, theta), (S[:, k], e_k))[1]

so no explicit Jacobian is formed; the augmented RHS is ``f(y)`` plus ``p``
JVPs of ``f`` (the JVP's primal yields ``f(y)`` for free).

This module implements the *augmented-system* form (one stock
:class:`diffrax.Kvaerno5` + :class:`diffrax.PIDController` whose error norm
covers ``S``, no ``dtmax`` cap). The per-step implicit solve uses the dense
augmented Jacobian; it removes the cap and is exact, and it is the right choice
for one or a few sensitivity parameters and for scalar-loss gradients. The
factorisation-sharing "simultaneous corrector" that additionally speeds up the
many-parameter case (reusing one ``(I - gamma.dt.J)`` factorisation across the
``S`` columns) is a further optimisation exposed through ``shared_factor`` and
not yet implemented.

References
----------
- Hindmarsh, A.C. et al. (2005). SUNDIALS: Suite of Nonlinear and
  Differential/Algebraic Equation Solvers. ACM TOMS 31(3), 363-396. (CVODES
  forward sensitivity; simultaneous vs staggered corrector.)
- Maly, T. & Petzold, L.R. (1996). Numerical methods and software for
  sensitivity analysis of DAE systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import diffrax
import jax
import jax.numpy as jnp

# Floor on the per-parameter error-control scale ``|theta_k|`` so a parameter
# that happens to pass through zero does not blow the sensitivity tolerance up
# to infinity. Far below any physical rate constant, so it never bites a
# nonzero parameter.
_SCALE_FLOOR = 1e-12


def resolve_sens_indices(network, sens_params) -> jnp.ndarray:
    """Resolve ``sens_params`` (names or indices) to a flat integer index array.

    Parameters
    ----------
    network : CompiledNetwork
        Provides ``param_index`` / ``parameters`` for name resolution.
    sens_params : sequence of str or int
        Free-parameter names (namespaced) or integer positions in ``params``.

    Returns
    -------
    jnp.ndarray
        Integer indices into the flat parameter vector, shape ``(k,)``.
    """
    if sens_params is None or len(sens_params) == 0:
        raise ValueError("sens_params must be a non-empty list of names or indices.")
    idx: list[int] = []
    for item in sens_params:
        if isinstance(item, str):
            if item not in network.param_index:
                raise KeyError(
                    f"Unknown parameter '{item}'. Available: {network.parameters}"
                )
            idx.append(network.param_index[item])
        else:
            i = int(item)
            if not (0 <= i < network.n_params):
                raise IndexError(
                    f"Sensitivity parameter index {i} out of range "
                    f"[0, {network.n_params})."
                )
            idx.append(i)
    return jnp.asarray(idx, dtype=jnp.int64 if jax.config.x64_enabled else jnp.int32)


def augmented_forward_sensitivity(
    f_flat: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    y0_flat: jnp.ndarray,
    params: jnp.ndarray,
    free_idx: jnp.ndarray,
    *,
    t0: float,
    t1: float,
    t_eval: Optional[jnp.ndarray],
    rtol: float,
    atol_y: jnp.ndarray,
    sens_rtol: Optional[float] = None,
    sens_atol=None,
    param_scale=None,
    dtmax: Optional[float] = None,
    max_steps: int = 1_000_000,
    shared_factor: bool = False,
    base_solver=None,
    factormax: Optional[float] = None,
):
    """Integrate ``z = [y; S]`` with adaptive control over both blocks.

    The primal carries no ``dtmax`` cap; the controller bounds the sensitivity
    error directly, so ``S`` is finite and exact where a capped ``jacfwd`` /
    ``jacrev`` would be needed.

    Parameters
    ----------
    f_flat : callable
        The reactor RHS as a flat map ``(t, y_flat, params) -> dy_flat``. It must
        depend on ``params`` everywhere they enter -- including any
        parameter-dependent stoichiometry -- so the JVP captures the full
        sensitivity. ``t`` is the integration variable (time, or axial position
        for a PFR); a spatially uniform reactor ignores it.
    y0_flat : jnp.ndarray
        Initial state, flat shape ``(ndof,)``.
    params : jnp.ndarray
        Full parameter vector, shape ``(n_params,)``.
    free_idx : jnp.ndarray
        Integer indices (shape ``(k,)``) of the sensitivity parameters within
        ``params``.
    t0, t1 : float
        Integration interval.
    t_eval : jnp.ndarray, optional
        Times/positions at which to record the solution. ``None`` records the
        endpoint only.
    rtol : float
        Relative tolerance on the state block ``y``.
    atol_y : jnp.ndarray
        Absolute tolerance on ``y``, shape ``(ndof,)``.
    sens_rtol : float, optional
        Relative tolerance on ``S``. Defaults to ``rtol`` (the CVODES default
        ``rtol_S = rtol``).
    sens_atol : float or array, optional
        Absolute tolerance on ``S``. Default follows CVODES:
        ``atol_S[:, k] = atol_y / scale_k`` with ``scale_k = |theta_k|`` -- so
        ``S`` is controlled to the same relative accuracy as ``y`` after scaling
        by the parameter magnitude. A scalar or ``(k,)`` value overrides it.
    param_scale : array, optional
        Per-parameter scale ``scale_k`` used in the default ``sens_atol``.
        Defaults to ``|theta_k|`` (floored). Ignored if ``sens_atol`` is given.
    dtmax : float, optional
        Maximum integrator step. ``None`` (default) -- the cap is not needed.
    max_steps : int, optional
        Maximum solver steps.
    shared_factor : bool, optional
        If ``True``, solve each stiff Newton step with the CVODES
        simultaneous-corrector (:class:`~aquakin.integrate._simultaneous_corrector.SimultaneousCorrector`):
        factorise the shared diagonal block ``D = I - gamma.dt.J`` once and
        forward-substitute across the ``S`` columns, instead of factorising the
        full ``n(1+k)`` augmented system. Exact (the Newton step is identical to
        the dense solve) and markedly cheaper for several parameters; for one or
        two parameters the dense ``False`` path is usually as fast or faster.

    Returns
    -------
    ts : jnp.ndarray
        Recorded times, shape ``(n_t,)``.
    y_traj : jnp.ndarray
        State trajectory, shape ``(n_t, ndof)``.
    S_traj : jnp.ndarray
        Sensitivity trajectory ``dy/dtheta``, shape ``(n_t, ndof, k)``.
    """
    ndof = int(y0_flat.shape[0])
    k = int(free_idx.shape[0])
    n_params = int(params.shape[0])

    # Unit parameter tangents: row j selects free parameter free_idx[j].
    E = jnp.zeros((k, n_params)).at[jnp.arange(k), free_idx].set(1.0)

    # Column-major augmented layout: z = [y (ndof); S_0 (ndof); ...; S_{k-1}].
    # Each sensitivity column is a contiguous ndof block, which is what makes the
    # block-arrow structure the simultaneous corrector exploits contiguous.
    def aug_rhs(t, z, args):
        p = args
        y = z[:ndof]
        S = z[ndof:].reshape(k, ndof)        # row j = sensitivity column j
        # Linearise f once at the shared point (y, theta): dy = f(y) (the primal,
        # computed once) and each sensitivity column is the same linear map
        # applied to (S_j, e_j). This is the variational RHS
        # dS_j/dt = J . S_j + f_theta_j without forming J.
        dy, f_jvp = jax.linearize(lambda yy, pp: f_flat(t, yy, pp), y, p)
        dS = jax.vmap(f_jvp, in_axes=(0, 0))(S, E)   # (k, ndof)
        return jnp.concatenate([dy, dS.reshape(-1)])

    z0 = jnp.concatenate([y0_flat, jnp.zeros(ndof * k)])

    # --- Error-control tolerances over the augmented state ----------------
    atol_y = jnp.broadcast_to(jnp.asarray(atol_y, dtype=float), (ndof,))
    theta = params[free_idx]
    if param_scale is None:
        scale = jnp.maximum(jnp.abs(theta), _SCALE_FLOOR)
    else:
        scale = jnp.broadcast_to(jnp.asarray(param_scale, dtype=float), (k,))

    # atol_S (k, ndof), row j = atol_y / scale_j -- column-major to match z.
    if sens_atol is None:
        atol_S = atol_y[None, :] / scale[:, None]          # (k, ndof)
    else:
        atol_S = jnp.broadcast_to(
            jnp.asarray(sens_atol, dtype=float), (k,)
        )[:, None] * jnp.ones((1, ndof))
    sens_rtol_v = rtol if sens_rtol is None else float(sens_rtol)

    atol_aug = jnp.concatenate([atol_y, atol_S.reshape(-1)])
    rtol_aug = jnp.concatenate(
        [jnp.full((ndof,), float(rtol)), jnp.full((ndof * k,), sens_rtol_v)]
    )

    sc = None
    if shared_factor:
        from aquakin.integrate._simultaneous_corrector import SimultaneousCorrector

        sc = SimultaneousCorrector(ndof=ndof, n_sens=k)

    if base_solver is None and factormax is None:
        # Reactor default: bare Kvaerno5 with the per-stage Newton tied to the
        # step controller's tolerances. Bit-identical to the historic
        # forward-sensitivity solve.
        from diffrax import VeryChord, with_stepsize_controller_tols

        if shared_factor:
            root_finder = with_stepsize_controller_tols(VeryChord)(linear_solver=sc)
            solver = diffrax.Kvaerno5(root_finder=root_finder)
        else:
            solver = diffrax.Kvaerno5()
        controller = diffrax.PIDController(rtol=rtol_aug, atol=atol_aug, dtmax=dtmax)
    else:
        # Enhanced path (the stiff plant): route the augmented [y; S] solve
        # through the single-source-of-truth solver helpers so it inherits the
        # decoupled Newton + factormax the plant's forward/adjoint solves use,
        # with a supplied lower-order base solver (Kvaerno3) and the block-arrow
        # SimultaneousCorrector for the per-stage linear algebra.
        from aquakin.integrate._common import (build_implicit_solver,
                                               build_step_controller)

        bs = diffrax.Kvaerno5() if base_solver is None else base_solver
        solver = build_implicit_solver(rtol, atol_aug, solver=bs,
                                       linear_solver=sc, force_root_finder=True)
        controller = build_step_controller(rtol_aug, atol_aug,
                                            factormax=factormax, dtmax=dtmax)

    saveat = (
        diffrax.SaveAt(t1=True)
        if t_eval is None
        else diffrax.SaveAt(ts=jnp.asarray(t_eval))
    )
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(aug_rhs),
        solver,
        t0=float(t0),
        t1=float(t1),
        dt0=None,
        y0=z0,
        args=params,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=max_steps,
    )
    ts = sol.ts
    y_traj = sol.ys[:, :ndof]
    # Column-major -> (n_t, ndof, k).
    S_traj = sol.ys[:, ndof:].reshape(sol.ys.shape[0], k, ndof).transpose(0, 2, 1)
    return ts, y_traj, S_traj


def build_jitted_sensitivity_solve(
    make_f_flat,
    free_idx: jnp.ndarray,
    *,
    t0: float,
    t1: float,
    has_t_eval: bool,
    rtol: float,
    atol_y: jnp.ndarray,
    sens_rtol,
    dtmax,
    max_steps: int,
    shared_factor: bool,
):
    """Build a ``jax.jit``-compiled forward-sensitivity solve for reuse.

    Mirrors the per-call-signature jit caching of ``reactor.solve`` so repeated
    evaluations (a calibration loop, an ensemble) compile once and then run
    without re-tracing -- which is what makes the simultaneous-corrector speedup
    visible in practice.

    ``make_f_flat(condition_arrays) -> f_flat`` rebuilds the flat RHS from the
    (traced) condition arrays, so condition overrides do not stale the cache.
    ``t_eval`` is passed as a runtime argument (not closed over), so different
    sample times of the same shape reuse the same compiled function.
    """
    if has_t_eval:
        @jax.jit
        def _solve(y0_flat, params, condition_arrays, t_eval):
            return augmented_forward_sensitivity(
                make_f_flat(condition_arrays), y0_flat, params, free_idx,
                t0=t0, t1=t1, t_eval=t_eval, rtol=rtol, atol_y=atol_y,
                sens_rtol=sens_rtol, dtmax=dtmax, max_steps=max_steps,
                shared_factor=shared_factor,
            )
        return _solve

    @jax.jit
    def _solve(y0_flat, params, condition_arrays):
        return augmented_forward_sensitivity(
            make_f_flat(condition_arrays), y0_flat, params, free_idx,
            t0=t0, t1=t1, t_eval=None, rtol=rtol, atol_y=atol_y,
            sens_rtol=sens_rtol, dtmax=dtmax, max_steps=max_steps,
            shared_factor=shared_factor,
        )
    return _solve


def run_forward_sensitivity(
    make_f_flat,
    y0_flat: jnp.ndarray,
    params: jnp.ndarray,
    free_idx: jnp.ndarray,
    cond,
    *,
    t0: float,
    t1: float,
    t_eval,
    rtol: float,
    atol_y: jnp.ndarray,
    sens_rtol,
    sens_atol,
    param_scale,
    dtmax,
    max_steps: int,
    shared_factor: bool,
    cache: dict,
    cache_key,
):
    """Run a forward-sensitivity solve and return ``(ts, y_traj, S_traj)``.

    The single dispatch shared by every reactor's ``solve_sensitivity`` (Batch /
    PFR / Biofilm): each reactor builds its own flat RHS factory, flattened state,
    error-control vector, time bounds, cache key and result wrapping; this routes
    them through the two paths uniformly so the dispatch cannot drift between
    reactors.

    Two paths, matching the per-call jit caching of ``reactor.solve``:

    - **Non-default sensitivity tolerances** (``sens_atol`` or ``param_scale``
      given) bypass the compile cache -- they would bloat the key -- and call
      :func:`augmented_forward_sensitivity` directly.
    - **The default path** is cached: a jitted solve from
      :func:`build_jitted_sensitivity_solve` is built once per ``cache_key`` and
      reused, then called with the (runtime) conditions and optional ``t_eval``.

    Parameters
    ----------
    make_f_flat : Callable
        ``condition_arrays -> (t, y_flat, p) -> dy_flat`` flat RHS factory.
    y0_flat : jnp.ndarray
        Flattened initial state.
    params : jnp.ndarray
        Parameter vector.
    free_idx : jnp.ndarray
        Indices of the sensitivity parameters (from
        :func:`resolve_sens_indices`).
    cond : dict
        The condition-field arrays.
    t0, t1, t_eval
        Integration bounds and save times (``t_eval`` ``None`` for solver-chosen).
    rtol, atol_y, sens_rtol, sens_atol, param_scale, dtmax, max_steps, shared_factor
        Solver / sensitivity controls, forwarded unchanged.
    cache : dict
        The reactor's ``_sens_jit_cache``.
    cache_key : hashable
        Per-call-signature key for the cached jitted solve.

    Returns
    -------
    (ts, y_traj, S_traj)
        Raw solver output; the caller reshapes / wraps it into its solution type.
    """
    if sens_atol is not None or param_scale is not None:
        return augmented_forward_sensitivity(
            make_f_flat(cond), y0_flat, params, free_idx,
            t0=t0, t1=t1, t_eval=t_eval, rtol=rtol, atol_y=atol_y,
            sens_rtol=sens_rtol, sens_atol=sens_atol, param_scale=param_scale,
            dtmax=dtmax, max_steps=max_steps, shared_factor=shared_factor,
        )
    jitted = cache.get(cache_key)
    if jitted is None:
        jitted = build_jitted_sensitivity_solve(
            make_f_flat, free_idx, t0=t0, t1=t1,
            has_t_eval=t_eval is not None, rtol=rtol, atol_y=atol_y,
            sens_rtol=sens_rtol, dtmax=dtmax, max_steps=max_steps,
            shared_factor=shared_factor,
        )
        cache[cache_key] = jitted
    if t_eval is None:
        return jitted(y0_flat, params, cond)
    return jitted(y0_flat, params, cond, t_eval)


@dataclass
class ForwardSensitivityResult:
    """Result of :func:`forward_sensitivity`.

    Attributes
    ----------
    solution : object
        The reactor solution (``BatchSolution`` / ``PFRSolution`` /
        ``BiofilmSolution``) -- the usual, uncapped state trajectory.
    S : jnp.ndarray
        Sensitivity ``d(output)/d(theta)`` at the saved times, shape
        ``(n_t, n_species, n_sens_params)``. For a biofilm this is the bulk
        (measurable) sensitivity, aligned with ``solution.C``.
    sens_params : list[str]
        The sensitivity-parameter names (resolved to namespaced names).
    network : object
        The compiled network (for the name accessors).
    """

    solution: Any
    S: jnp.ndarray
    sens_params: list[str]
    network: Any

    def S_named(self, species: str) -> jnp.ndarray:
        """Sensitivity of one species over time, shape ``(n_t, n_sens_params)``."""
        if species not in self.network.species_index:
            raise KeyError(
                f"Unknown species '{species}'. Available: {self.network.species}"
            )
        return self.S[:, self.network.species_index[species], :]

    def dC_dparam(self, species: str, param: str) -> jnp.ndarray:
        """Sensitivity of one species w.r.t. one parameter, shape ``(n_t,)``."""
        if param not in self.sens_params:
            raise KeyError(
                f"'{param}' is not a sensitivity parameter. Available: "
                f"{self.sens_params}"
            )
        j = self.sens_params.index(param)
        return self.S_named(species)[:, j]


def forward_sensitivity(
    reactor,
    C0: jnp.ndarray,
    params: jnp.ndarray,
    *,
    sens_params,
    **solve_kwargs,
) -> ForwardSensitivityResult:
    """Forward-sensitivity solve through any reactor exposing ``solve_sensitivity``.

    Thin wrapper mirroring :func:`aquakin.sensitivity`: it calls the reactor's
    :meth:`solve_sensitivity` and packages the ``(solution, S)`` pair with the
    resolved parameter names and convenience accessors.

    Parameters
    ----------
    reactor : BatchReactor, PlugFlowReactor or BiofilmReactor
        Any reactor with a ``solve_sensitivity`` method.
    C0 : jnp.ndarray
        Initial state.
    params : jnp.ndarray
        Full parameter vector.
    sens_params : list of str or int
        Sensitivity-parameter names or indices.
    **solve_kwargs
        Passed through to ``reactor.solve_sensitivity`` (e.g. ``t_span``,
        ``t_eval``, ``conditions``, ``sens_rtol``, ``sens_atol``,
        ``param_scale``).

    Returns
    -------
    ForwardSensitivityResult
    """
    network = reactor.network
    free_idx = resolve_sens_indices(network, sens_params)
    names = [network.parameters[int(i)] for i in free_idx]
    sol, S = reactor.solve_sensitivity(
        C0, params, sens_params=sens_params, **solve_kwargs
    )
    return ForwardSensitivityResult(
        solution=sol, S=S, sens_params=names, network=network
    )
