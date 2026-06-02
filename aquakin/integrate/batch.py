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
    _run_diffeqsolve,
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
        uncapped, which is fastest for forward solves. Set it when
        *differentiating* a stiff network: an L-stable solver can step over
        the fastest reaction timescale and damp those modes in the primal,
        but their sensitivity is then ill-resolved and ``jax.grad`` /
        ``jax.jvp`` return non-finite values. Capping ``dtmax`` to a small
        multiple of the fastest reaction timescale fixes it (both AD modes)
        and the gradients match finite differences.
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
    ) -> None:
        conditions.validate_required(network.conditions_required)
        self.network = network
        self.conditions = conditions
        self.rtol = rtol
        self.atol = _coerce_atol(atol, network.n_species)
        self.adjoint = adjoint
        self.dtmax = dtmax
        # Cache jit-compiled inner solve keyed on (t0, t1, t_eval_shape).
        # First call with a new signature pays the trace cost; subsequent
        # calls reuse the compiled graph.
        self._jit_cache: dict = {}

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

    def _build_jitted_solve(self, t0: float, t1: float, has_t_eval: bool):
        """Build a jit-compiled inner solver for a specific call signature."""
        network = self.network
        rtol = self.rtol
        atol = self.atol
        adjoint = self.adjoint
        dtmax = self.dtmax

        if has_t_eval:
            @jax.jit
            def _solve(C0, params, condition_arrays, t_eval):
                # Hoist stoichiometry out of the per-step RHS so dynamic
                # (parameter-dependent) coefficients are evaluated once,
                # not on every ODE step.
                stoich = network.compute_stoich(params)

                def rhs(t, C, args):
                    return network.dCdt(C, args, condition_arrays, 0, stoich=stoich)

                sol = _run_diffeqsolve(
                    rhs,
                    t0=t0,
                    t1=t1,
                    y0=C0,
                    args=params,
                    saveat=diffrax.SaveAt(ts=t_eval),
                    rtol=rtol,
                    atol=atol,
                    adjoint=adjoint,
                    dtmax=dtmax,
                )
                return sol.ts, sol.ys

            return _solve

        @jax.jit
        def _solve(C0, params, condition_arrays):
            stoich = network.compute_stoich(params)

            def rhs(t, C, args):
                return network.dCdt(C, args, condition_arrays, 0, stoich=stoich)

            sol = _run_diffeqsolve(
                rhs,
                t0=t0,
                t1=t1,
                y0=C0,
                args=params,
                saveat=diffrax.SaveAt(t1=True),
                rtol=rtol,
                atol=atol,
                adjoint=adjoint,
                dtmax=dtmax,
            )
            return sol.ts, sol.ys

        return _solve
