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

        n_loc = max(conditions.n_locations, 1)
        if n_loc == 1:
            self._x_grid = jnp.asarray([0.0])
        else:
            self._x_grid = jnp.linspace(0.0, self.length, n_loc)
        # PFR solve takes no varying signature args, so we only ever need one
        # jitted variant; build it lazily on first solve.
        self._jitted_solve = None

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
            )
            return sol.ts, sol.ys

        return _solve
