"""Plug-flow reactor (1-D, steady-state) integrator."""

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
    _interp_fields_to_scalar,
    _run_diffeqsolve,
)


@dataclass
class PFRSolution(_HasNamedSpecies):
    """
    Solution returned by :meth:`PlugFlowReactor.solve`.

    Attributes
    ----------
    x : jnp.ndarray
        Axial positions at which the solution was recorded, shape ``(n_points,)``.
    C : jnp.ndarray
        Concentration profile, shape ``(n_points, n_species)``.
    network : CompiledNetwork
        Network used to produce this solution.
    """

    x: jnp.ndarray
    C: jnp.ndarray
    network: CompiledNetwork


class PlugFlowReactor:
    """
    Steady-state plug-flow reactor.

    The chemistry RHS is integrated along the reactor axis:

    .. math:: \\frac{dC}{dx} = \\frac{1}{v}\\, \\mathbf{S}^T\\, r(C, p, \\theta(x))

    Condition fields supplied at ``n_locations`` grid points are linearly
    interpolated to the integrator's current ``x``.

    Parameters
    ----------
    network : CompiledNetwork
    conditions : SpatialConditions
        Either a uniform single-location conditions object, or a spatially
        resolved one with ``n_locations >= 2``. Grid points are assumed evenly
        spaced over ``[0, length]``.
    n_points : int
        Number of axial output points at which to record the solution.
    length : float
        Reactor length.
    velocity : float
        Bulk velocity through the reactor.
    rtol : float, optional
        Relative tolerance for the ODE solver.
    atol : float or jnp.ndarray, optional
        Absolute tolerance, scalar or shape ``(n_species,)``. See
        :class:`BatchReactor` for the per-species rationale.
    """

    def __init__(
        self,
        network: CompiledNetwork,
        conditions: SpatialConditions,
        n_points: int,
        length: float,
        velocity: float,
        *,
        rtol: float = 1e-6,
        atol=1e-9,
        dtmax: Optional[float] = None,
        max_steps: int = 100_000,
    ) -> None:
        conditions.validate_required(network.conditions_required)
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if velocity <= 0:
            raise ValueError(f"velocity must be positive, got {velocity}")
        self.network = network
        self.conditions = conditions
        self.n_points = int(n_points)
        self.length = float(length)
        self.velocity = float(velocity)
        self.rtol = rtol
        self.atol = _coerce_atol(atol, network.n_species)
        self.dtmax = dtmax
        self.max_steps = int(max_steps)

        n_loc = max(conditions.n_locations, 1)
        if n_loc == 1:
            self._x_grid = jnp.asarray([0.0])
        else:
            self._x_grid = jnp.linspace(0.0, self.length, n_loc)
        # PFR solve takes no varying signature args, so we only ever need one
        # jitted variant; build it lazily on first solve.
        self._jitted_solve = None
        self._sens_jit_cache: dict = {}

    def solve(
        self,
        C0: jnp.ndarray,
        params: jnp.ndarray,
        *,
        conditions: Optional[SpatialConditions] = None,
    ) -> PFRSolution:
        """
        Integrate the steady-state PFR.

        Parameters
        ----------
        C0 : jnp.ndarray
            Inlet concentration vector, shape ``(n_species,)``.
        params : jnp.ndarray
            Rate constant vector, shape ``(n_params,)``.
        conditions : SpatialConditions, optional
            Override the conditions stored on the reactor for this call. Must
            match the constructor-time ``n_locations`` so the precomputed
            ``x_grid`` remains valid.

        Returns
        -------
        PFRSolution
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

        active_conditions = conditions if conditions is not None else self.conditions
        if active_conditions.n_locations != self.conditions.n_locations:
            raise ValueError(
                f"conditions override must have n_locations="
                f"{self.conditions.n_locations}, got {active_conditions.n_locations}."
            )

        fields = active_conditions.fields

        if self._jitted_solve is None:
            self._jitted_solve = self._build_jitted_solve()
        ts, ys = self._jitted_solve(C0, params, fields)
        return PFRSolution(x=ts, C=ys, network=self.network)

    def solve_sensitivity(
        self,
        C0: jnp.ndarray,
        params: jnp.ndarray,
        *,
        sens_params,
        conditions: Optional[SpatialConditions] = None,
        sens_rtol: Optional[float] = None,
        sens_atol=None,
        param_scale=None,
        shared_factor: Optional[bool] = None,
    ) -> tuple["PFRSolution", jnp.ndarray]:
        """Solve and return the forward sensitivity ``dC/dtheta`` along the axis.

        Integrates the augmented ``[C; S]`` system over the reactor length with
        adaptive control over both, so the sensitivity profile is exact and
        finite without a ``dtmax`` cap (see
        :mod:`aquakin.integrate.forward_sensitivity`).

        Parameters
        ----------
        C0, params, conditions
            As for :meth:`solve`.
        sens_params : list of str or int
            Namespaced parameter names (or integer indices into ``params``).
        sens_rtol, sens_atol, param_scale
            Sensitivity error-control tolerances (CVODES defaults).
        shared_factor : bool, optional
            CVODES simultaneous-corrector linear solve. ``None`` (default)
            auto-selects ``True`` for more than one sensitivity parameter, else
            ``False``.

        Returns
        -------
        sol : PFRSolution
            The usual axial concentration profile.
        S : jnp.ndarray
            Sensitivity ``dC/dtheta`` at the axial output points, shape
            ``(n_points, n_species, n_sens_params)``.
        """
        from aquakin.integrate._common import _interp_fields_to_scalar
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
        active = conditions if conditions is not None else self.conditions
        if active.n_locations != self.conditions.n_locations:
            raise ValueError(
                f"conditions override must have n_locations="
                f"{self.conditions.n_locations}, got {active.n_locations}."
            )

        free_idx = resolve_sens_indices(self.network, sens_params)
        if shared_factor is None:
            shared_factor = free_idx.shape[0] > 1
        fields = active.fields
        network = self.network
        velocity = self.velocity
        x_grid = self._x_grid
        single_loc = self.conditions.n_locations <= 1
        x_eval = jnp.linspace(0.0, self.length, self.n_points)
        atol_y = jnp.broadcast_to(jnp.asarray(self.atol, dtype=float),
                                  (network.n_species,))

        def make_f_flat(cond_arrays):
            def f_flat(x, C, p):
                cond = (
                    cond_arrays if single_loc
                    else _interp_fields_to_scalar(x, x_grid, cond_arrays)
                )
                return network.dCdt(C, p, cond, 0) / velocity
            return f_flat

        if sens_atol is not None or param_scale is not None:
            xs, y_traj, S_traj = augmented_forward_sensitivity(
                make_f_flat(fields), C0, params, free_idx,
                t0=0.0, t1=self.length, t_eval=x_eval,
                rtol=self.rtol, atol_y=atol_y,
                sens_rtol=sens_rtol, sens_atol=sens_atol, param_scale=param_scale,
                dtmax=self.dtmax, shared_factor=shared_factor,
            )
            return PFRSolution(x=xs, C=y_traj, network=network), S_traj

        cache_key = (
            tuple(int(i) for i in free_idx), bool(shared_factor),
            None if sens_rtol is None else float(sens_rtol),
        )
        jitted = self._sens_jit_cache.get(cache_key)
        if jitted is None:
            jitted = build_jitted_sensitivity_solve(
                make_f_flat, free_idx, t0=0.0, t1=self.length, has_t_eval=True,
                rtol=self.rtol, atol_y=atol_y, sens_rtol=sens_rtol,
                # The augmented [y; S] solve resolves the sensitivity transient
                # and is step-hungrier than the primal; use the same 1e6 budget
                # as BatchReactor (was 1e5, an inconsistency).
                dtmax=self.dtmax, max_steps=1_000_000, shared_factor=shared_factor,
            )
            self._sens_jit_cache[cache_key] = jitted

        xs, y_traj, S_traj = jitted(C0, params, fields, x_eval)
        return PFRSolution(x=xs, C=y_traj, network=network), S_traj

    def _build_jitted_solve(self):
        """Build a jit-compiled inner PFR solver. Single signature suffices."""
        network = self.network
        velocity = self.velocity
        x_grid = self._x_grid
        single_loc = self.conditions.n_locations <= 1
        length = self.length
        rtol = self.rtol
        atol = self.atol
        dtmax = self.dtmax
        max_steps = self.max_steps
        x_eval = jnp.linspace(0.0, length, self.n_points)

        @jax.jit
        def _solve(C0, params, fields):
            # Hoist parameter-dependent stoichiometry out of the per-step RHS.
            stoich = network.compute_stoich(params)

            def rhs(x, C, args):
                params_ = args
                if single_loc:
                    cond = fields
                else:
                    cond = _interp_fields_to_scalar(x, x_grid, fields)
                return network.dCdt(C, params_, cond, 0, stoich=stoich) / velocity

            sol = _run_diffeqsolve(
                rhs,
                t0=0.0,
                t1=length,
                y0=C0,
                args=params,
                saveat=diffrax.SaveAt(ts=x_eval),
                rtol=rtol,
                atol=atol,
                dtmax=dtmax,
                max_steps=max_steps,
            )
            return sol.ts, sol.ys

        return _solve
