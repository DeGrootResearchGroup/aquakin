"""Batch (0-D) reactor: integrate chemistry at a single spatial location."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import diffrax
import jax
import jax.numpy as jnp

from aquakin.core.conditions import SpatialConditions
from aquakin.core.network import CompiledNetwork
from aquakin.integrate._common import (
    _HasNamedSpecies,
    _coerce_atol,
    solve_chemistry,
    validate_t_eval,
)


@dataclass
class BatchSolution(_HasNamedSpecies):
    """
    Solution returned by :meth:`BatchReactor.solve`.

    Attributes
    ----------
    t : jnp.ndarray
        Times at which the solution was recorded, shape ``(n_t,)``.
    C : jnp.ndarray
        Concentration trajectory, shape ``(n_t, n_species)``.
    network : CompiledNetwork
        The network that produced this solution. Retained so that the
        inherited :meth:`C_named` can look up species by name.
    """

    t: jnp.ndarray
    C: jnp.ndarray
    network: CompiledNetwork


class BatchReactor:
    """
    Stateless 0-D (batch) reactor.

    Parameters
    ----------
    network : CompiledNetwork
        Compiled reaction network.
    conditions : SpatialConditions
        Condition fields. For a batch reactor only the single location
        ``loc_idx=0`` is used.
    rtol : float, optional
        Relative tolerance for the ODE solver.
    atol : float or jnp.ndarray, optional
        Absolute tolerance for the ODE solver. If an array, it must have
        shape ``(n_species,)`` and gives a per-species tolerance. Useful when
        some species (e.g. radical intermediates) sit several orders of
        magnitude below the bulk concentrations.
    adjoint : diffrax.AbstractAdjoint, optional
        Adjoint strategy. Defaults to
        :class:`diffrax.RecursiveCheckpointAdjoint`, the right choice for
        reverse-mode parameter estimation: its memory grows only
        logarithmically with the number of steps (binomial checkpointing),
        so it is preferred for long or stiff integrations and for
        ``jax.grad``-based fitting.

        Pass ``diffrax.DirectAdjoint()`` only when you need **forward-mode**
        autodiff (``jax.jvp`` / ``jax.jacfwd``) through the solve --- for
        example a forward-mode sensitivity Jacobian or a Gauss-Newton /
        Fisher information matrix. ``RecursiveCheckpointAdjoint`` registers a
        ``custom_vjp`` (a reverse-mode rule only), so forward-mode is rejected
        with "can't apply forward-mode autodiff (jvp) to a custom_vjp
        function"; ``DirectAdjoint`` is plainly differentiable in both modes.
        Its drawback is memory: it stores/unrolls the whole solve (cost grows
        with the number of steps), so reserve it for short integrations or
        when forward-mode is actually required. Either way, cap ``dtmax`` when
        differentiating a stiff network (see below).
    dtmax : float, optional
        Maximum integrator step size. ``None`` (default) leaves the step
        uncapped, which is fastest for forward solves. Set it for
        *reverse-mode* differentiation of a stiff network: an L-stable solver
        can step over the fastest reaction timescale and damp those modes in
        the primal (which stays accurate). Forward mode (``jax.jvp`` /
        ``jax.jacfwd``) then stays finite at any step, but reverse mode
        (``jax.grad``) returns non-finite values above a step-size threshold
        (a backward-accumulation overflow set by the per-step stiffness).
        Capping ``dtmax`` to a small multiple of the fastest reaction
        timescale restores a finite reverse gradient that matches forward mode
        and finite differences.
    max_steps : int, optional
        Maximum number of internal solver steps (default 100000). Raise it for
        long or very stiff forward solves that exhaust the default budget.
    """

    def __init__(
        self,
        network: CompiledNetwork,
        conditions: SpatialConditions,
        *,
        rtol: float = 1e-6,
        atol=1e-9,
        adjoint: Optional[diffrax.AbstractAdjoint] = None,
        dtmax: Optional[float] = None,
        max_steps: int = 100_000,
    ) -> None:
        conditions.validate_required(network.conditions_required)
        self.network = network
        self.conditions = conditions
        self.rtol = rtol
        self.atol = _coerce_atol(atol, network.n_species)
        self.adjoint = adjoint
        self.dtmax = dtmax
        self.max_steps = int(max_steps)
        # Cache jit-compiled inner solve keyed on (t0, t1, t_eval_shape).
        # First call with a new signature pays the trace cost; subsequent
        # calls reuse the compiled graph.
        self._jit_cache: dict = {}
        # Separate cache for the forward-sensitivity solves.
        self._sens_jit_cache: dict = {}

    def solve(
        self,
        C0: jnp.ndarray,
        params: jnp.ndarray,
        t_span: tuple[float, float],
        t_eval: Optional[jnp.ndarray] = None,
        *,
        conditions: Optional[SpatialConditions] = None,
    ) -> BatchSolution:
        """
        Integrate the reaction network over a time span.

        Parameters
        ----------
        C0 : jnp.ndarray
            Initial concentration vector, shape ``(n_species,)``.
        params : jnp.ndarray
            Rate constant vector, shape ``(n_params,)``.
        t_span : tuple of float
            ``(t_start, t_end)`` integration interval.
        t_eval : jnp.ndarray, optional
            Time points at which to record solution. If ``None`` the solver
            returns endpoints only.
        conditions : SpatialConditions, optional
            Override the conditions stored on the reactor for this call. Used
            by :func:`aquakin.sensitivity` to differentiate through condition
            fields without mutating the reactor.

        Returns
        -------
        BatchSolution
        """
        C0 = jnp.asarray(C0)
        params = jnp.asarray(params)
        if C0.shape != (self.network.n_species,):
            raise ValueError(
                f"C0 has shape {C0.shape}, expected ({self.network.n_species},)"
            )
        if params.shape != (self.network.n_params,):
            raise ValueError(
                f"params has shape {params.shape}, expected ({self.network.n_params},)"
            )

        t0, t1 = float(t_span[0]), float(t_span[1])
        if not (t1 > t0):
            raise ValueError(
                f"t_span end must exceed start; got ({t0}, {t1})."
            )
        active_conditions = conditions if conditions is not None else self.conditions
        condition_arrays = active_conditions.fields

        if t_eval is None:
            t_eval_arr = None
            cache_key = (t0, t1, None)
        else:
            t_eval_arr = jnp.asarray(t_eval)
            self._validate_t_eval(t_eval_arr, t0, t1)
            cache_key = (t0, t1, tuple(t_eval_arr.shape))

        jitted = self._jit_cache.get(cache_key)
        if jitted is None:
            jitted = self._build_jitted_solve(t0, t1, t_eval_arr is not None)
            self._jit_cache[cache_key] = jitted

        if t_eval_arr is None:
            ts, ys = jitted(C0, params, condition_arrays)
        else:
            ts, ys = jitted(C0, params, condition_arrays, t_eval_arr)
        return BatchSolution(t=ts, C=ys, network=self.network)

    def solve_sensitivity(
        self,
        C0: jnp.ndarray,
        params: jnp.ndarray,
        t_span: tuple[float, float],
        t_eval: Optional[jnp.ndarray] = None,
        *,
        sens_params,
        conditions: Optional[SpatialConditions] = None,
        sens_rtol: Optional[float] = None,
        sens_atol=None,
        param_scale=None,
        shared_factor: Optional[bool] = None,
    ) -> tuple["BatchSolution", jnp.ndarray]:
        """Solve and return the forward sensitivity ``dC/dtheta`` alongside ``C``.

        Integrates the augmented ``[C; S]`` system (state plus sensitivity) with
        adaptive control over both, so the sensitivity is exact and finite
        without the ``dtmax`` cap that ordinary AD through a stiff solve needs
        (see :mod:`aquakin.integrate.forward_sensitivity`).

        Parameters
        ----------
        C0, params, t_span, t_eval, conditions
            As for :meth:`solve`.
        sens_params : list of str or int
            Namespaced parameter names (or integer indices into ``params``) to
            differentiate with respect to.
        sens_rtol, sens_atol, param_scale
            Sensitivity error-control tolerances. Defaults follow CVODES:
            ``rtol_S = rtol`` and ``atol_S = atol / |theta_k|``. See
            :func:`~aquakin.integrate.forward_sensitivity.augmented_forward_sensitivity`.
        shared_factor : bool, optional
            Use the CVODES simultaneous-corrector linear solve (factorise the
            shared diagonal block once, forward-substitute across the
            sensitivity columns). ``None`` (default) auto-selects: ``True`` for
            more than one sensitivity parameter (where it is markedly cheaper
            than the dense augmented solve), ``False`` for a single parameter.

        Returns
        -------
        sol : BatchSolution
            The usual state trajectory (uncapped, exact).
        S : jnp.ndarray
            Sensitivity ``dC/dtheta`` at the saved times, shape
            ``(n_t, n_species, n_sens_params)``.
        """
        from aquakin.integrate.forward_sensitivity import (
            augmented_forward_sensitivity,
            build_jitted_sensitivity_solve,
            resolve_sens_indices,
        )

        C0 = jnp.asarray(C0)
        params = jnp.asarray(params)
        if C0.shape != (self.network.n_species,):
            raise ValueError(
                f"C0 has shape {C0.shape}, expected ({self.network.n_species},)"
            )
        if params.shape != (self.network.n_params,):
            raise ValueError(
                f"params has shape {params.shape}, expected ({self.network.n_params},)"
            )
        t0, t1 = float(t_span[0]), float(t_span[1])
        if not (t1 > t0):
            raise ValueError(f"t_span end must exceed start; got ({t0}, {t1}).")

        free_idx = resolve_sens_indices(self.network, sens_params)
        if shared_factor is None:
            shared_factor = free_idx.shape[0] > 1
        active = conditions if conditions is not None else self.conditions
        cond = active.fields
        network = self.network
        atol_y = jnp.broadcast_to(jnp.asarray(self.atol, dtype=float),
                                  (network.n_species,))
        t_eval_arr = None if t_eval is None else jnp.asarray(t_eval)
        if t_eval_arr is not None:
            self._validate_t_eval(t_eval_arr, t0, t1)

        # Non-default sensitivity tolerances bypass the compile cache (they would
        # bloat the key); the common default path is cached like ``solve``.
        if sens_atol is not None or param_scale is not None:
            def f_flat(t, y, p):
                return network.dCdt(y, p, cond, 0)

            ts, y_traj, S_traj = augmented_forward_sensitivity(
                f_flat, C0, params, free_idx,
                t0=t0, t1=t1, t_eval=t_eval_arr,
                rtol=self.rtol, atol_y=atol_y,
                sens_rtol=sens_rtol, sens_atol=sens_atol, param_scale=param_scale,
                dtmax=self.dtmax, shared_factor=shared_factor,
            )
            return BatchSolution(t=ts, C=y_traj, network=network), S_traj

        cache_key = (
            t0, t1, None if t_eval_arr is None else tuple(t_eval_arr.shape),
            tuple(int(i) for i in free_idx), bool(shared_factor),
            None if sens_rtol is None else float(sens_rtol),
        )
        jitted = self._sens_jit_cache.get(cache_key)
        if jitted is None:
            def make_f_flat(condition_arrays):
                return lambda t, y, p: network.dCdt(y, p, condition_arrays, 0)

            jitted = build_jitted_sensitivity_solve(
                make_f_flat, free_idx, t0=t0, t1=t1,
                has_t_eval=t_eval_arr is not None, rtol=self.rtol, atol_y=atol_y,
                sens_rtol=sens_rtol, dtmax=self.dtmax, max_steps=1_000_000,
                shared_factor=shared_factor,
            )
            self._sens_jit_cache[cache_key] = jitted

        if t_eval_arr is None:
            ts, y_traj, S_traj = jitted(C0, params, cond)
        else:
            ts, y_traj, S_traj = jitted(C0, params, cond, t_eval_arr)
        return BatchSolution(t=ts, C=y_traj, network=network), S_traj

    _validate_t_eval = staticmethod(validate_t_eval)

    def _build_jitted_solve(self, t0: float, t1: float, has_t_eval: bool):
        """Build a jit-compiled inner solver for a specific call signature."""
        network = self.network
        kw = dict(
            t0=t0, t1=t1, rtol=self.rtol, atol=self.atol,
            adjoint=self.adjoint, dtmax=self.dtmax, max_steps=self.max_steps,
        )

        if has_t_eval:
            @jax.jit
            def _solve(C0, params, condition_arrays, t_eval):
                sol = solve_chemistry(
                    network, C0, params,
                    cond_fn=lambda t: condition_arrays,
                    saveat=diffrax.SaveAt(ts=t_eval), **kw,
                )
                return sol.ts, sol.ys

            return _solve

        @jax.jit
        def _solve(C0, params, condition_arrays):
            sol = solve_chemistry(
                network, C0, params,
                cond_fn=lambda t: condition_arrays,
                saveat=diffrax.SaveAt(t1=True), **kw,
            )
            return sol.ts, sol.ys

        return _solve
