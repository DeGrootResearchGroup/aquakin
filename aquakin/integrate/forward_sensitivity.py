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
        If ``True``, use the CVODES simultaneous-corrector factorisation sharing
        across the ``S`` columns. Not yet implemented (raises).

    Returns
    -------
    ts : jnp.ndarray
        Recorded times, shape ``(n_t,)``.
    y_traj : jnp.ndarray
        State trajectory, shape ``(n_t, ndof)``.
    S_traj : jnp.ndarray
        Sensitivity trajectory ``dy/dtheta``, shape ``(n_t, ndof, k)``.
    """
    if shared_factor:
        raise NotImplementedError(
            "shared_factor=True (the CVODES simultaneous-corrector factorisation "
            "sharing across sensitivity columns) is not yet implemented. Use "
            "shared_factor=False, which integrates the augmented [y; S] system "
            "with a dense implicit step -- exact and cap-free, and the right "
            "choice for one or a few sensitivity parameters."
        )

    ndof = int(y0_flat.shape[0])
    k = int(free_idx.shape[0])
    n_params = int(params.shape[0])

    # Unit parameter tangents: row j selects free parameter free_idx[j].
    E = jnp.zeros((k, n_params)).at[jnp.arange(k), free_idx].set(1.0)

    def aug_rhs(t, z, args):
        p = args
        y = z[:ndof]
        S = z[ndof:].reshape(ndof, k)
        # Linearise f once at the shared point (y, theta): dy = f(y) (the primal,
        # computed once) and each sensitivity column is the same linear map
        # applied to (S[:, j], e_j). This is the variational RHS
        # dS[:, j]/dt = J . S[:, j] + f_theta[:, j] without forming J.
        dy, f_jvp = jax.linearize(lambda yy, pp: f_flat(t, yy, pp), y, p)
        dS = jax.vmap(f_jvp, in_axes=(1, 0))(S, E).T   # (ndof, k)
        return jnp.concatenate([dy, dS.reshape(-1)])

    z0 = jnp.concatenate([y0_flat, jnp.zeros(ndof * k)])

    # --- Error-control tolerances over the augmented state ----------------
    atol_y = jnp.broadcast_to(jnp.asarray(atol_y, dtype=float), (ndof,))
    theta = params[free_idx]
    if param_scale is None:
        scale = jnp.maximum(jnp.abs(theta), _SCALE_FLOOR)
    else:
        scale = jnp.broadcast_to(jnp.asarray(param_scale, dtype=float), (k,))

    if sens_atol is None:
        atol_S = atol_y[:, None] / scale[None, :]          # (ndof, k)
    else:
        atol_S = jnp.broadcast_to(
            jnp.asarray(sens_atol, dtype=float), (k,)
        )[None, :] * jnp.ones((ndof, 1))
    sens_rtol_v = rtol if sens_rtol is None else float(sens_rtol)

    atol_aug = jnp.concatenate([atol_y, atol_S.reshape(-1)])
    rtol_aug = jnp.concatenate(
        [jnp.full((ndof,), float(rtol)), jnp.full((ndof * k,), sens_rtol_v)]
    )

    saveat = (
        diffrax.SaveAt(t1=True)
        if t_eval is None
        else diffrax.SaveAt(ts=jnp.asarray(t_eval))
    )
    controller = diffrax.PIDController(rtol=rtol_aug, atol=atol_aug, dtmax=dtmax)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(aug_rhs),
        diffrax.Kvaerno5(),
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
    S_traj = sol.ys[:, ndof:].reshape(sol.ys.shape[0], ndof, k)
    return ts, y_traj, S_traj


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
