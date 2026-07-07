"""Plug-flow reactor (1-D, steady-state) integrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import diffrax
import jax
import jax.numpy as jnp

from aquakin.core.conditions import SpatialConditions
from aquakin.core.model import CompiledModel
from aquakin.integrate._common import (
    DifferentiationConfig,
    ExportableSolutionMixin,
    GradientCheckMixin,
    IntegratorConfig,
    PlottableSolutionMixin,
    _HasNamedSpecies,
    _interp_fields_to_scalar,
    cached_jitted_solver,
    friendly_solve_errors,
    init_solver_settings,
    reactor_settings_key,
    resolve_state_atol,
    solve_chemistry,
    validate_C0_params,
)


@dataclass
class PFRSolution(_HasNamedSpecies, PlottableSolutionMixin, ExportableSolutionMixin):
    """
    Solution returned by :meth:`PlugFlowReactor.solve`.

    Attributes
    ----------
    x : jnp.ndarray
        Axial positions at which the solution was recorded, shape ``(n_points,)``.
    C : jnp.ndarray
        Concentration profile, shape ``(n_points, n_species)``. This is the raw
        integrated state. If the model sets ``clip_negative_states``, individual
        entries may be **small transient negatives**: the ``max(C, 0)`` clamp is
        applied only when evaluating the reaction rates (and state-derived
        conditions), not to the saved state. These are a normal numerical
        transient, not an error; clip with ``jnp.maximum(sol.C, 0.0)`` for display
        if needed.
    model : CompiledModel
        Model used to produce this solution.
    """

    x: jnp.ndarray
    C: jnp.ndarray
    model: CompiledModel

    def _table_index(self):
        # A PFR profile is indexed by axial position, not time.
        return "x", self.x

    def _independent_axis_label(self) -> str:
        return "axial position [m]"


class PlugFlowReactor(GradientCheckMixin):
    """
    Steady-state plug-flow reactor.

    The chemistry RHS is integrated along the reactor axis:

    .. math:: \\frac{dC}{dx} = \\frac{1}{v}\\, \\mathbf{S}^T\\, r(C, p, \\theta(x))

    Condition fields supplied at ``n_locations`` grid points are linearly
    interpolated to the integrator's current ``x``.

    Parameters
    ----------
    model : CompiledModel
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
        Absolute tolerance, scalar or shape ``(n_species,)``. Defaults to
        ``None`` -> a per-component noise floor scaled off the model reference
        concentrations (see :class:`BatchReactor` for the per-species rationale).
    integrator : IntegratorConfig, optional
        Integrator / step-size configuration (ESDIRK ``order``, ``factormax``,
        ``dtmax``, ``max_steps``, an explicit ``solver``). See
        :class:`BatchReactor`. Set ``dtmax`` for reverse-mode differentiation of
        a stiff model.
    diff : DifferentiationConfig, optional
        Autodiff configuration (``mode``, ``method``). See :class:`BatchReactor`.
    """

    def __init__(
        self,
        model: CompiledModel,
        conditions: SpatialConditions,
        n_points: int,
        length: float,
        velocity: float,
        *,
        rtol: float = 1e-6,
        atol=None,
        integrator: IntegratorConfig = IntegratorConfig(),
        diff: DifferentiationConfig = DifferentiationConfig(),
    ) -> None:
        conditions.validate_required(model.conditions_required)
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if velocity <= 0:
            raise ValueError(f"velocity must be positive, got {velocity}")
        init_solver_settings(self, model, rtol=rtol, integrator=integrator, diff=diff)
        self.conditions = conditions
        self.n_points = int(n_points)
        self.length = float(length)
        self.velocity = float(velocity)
        self.atol = resolve_state_atol(model, atol)

        self.n_locations = max(conditions.n_locations, 1)
        if self.n_locations == 1:
            self._x_grid = jnp.asarray([0.0])
        else:
            self._x_grid = jnp.linspace(0.0, self.length, self.n_locations)
        self._sens_jit_cache: dict = {}

    def solve(
        self,
        C0: jnp.ndarray,
        *,
        params: Optional[jnp.ndarray] = None,
        conditions: Optional[SpatialConditions] = None,
    ) -> PFRSolution:
        """
        Integrate the steady-state PFR.

        Parameters
        ----------
        C0 : jnp.ndarray
            Inlet concentration vector, shape ``(n_species,)``.
        params : jnp.ndarray, optional
            Rate constant vector, shape ``(n_params,)``. Defaults to
            ``model.default_parameters()``.
        conditions : SpatialConditions, optional
            Override the conditions stored on the reactor for this call. Must
            match the constructor-time ``n_locations`` so the precomputed
            ``x_grid`` remains valid.

        Returns
        -------
        PFRSolution
        """
        C0 = jnp.asarray(C0)
        params = self.model.default_parameters() if params is None else jnp.asarray(params)
        validate_C0_params(self.model, C0, params)

        active_conditions = conditions if conditions is not None else self.conditions
        if active_conditions.n_locations != self.conditions.n_locations:
            raise ValueError(
                f"conditions override must have n_locations="
                f"{self.conditions.n_locations}, got {active_conditions.n_locations}."
            )

        fields = active_conditions.fields

        # Shared across reactor instances: a reactor with the same model,
        # solver settings and geometry (velocity / length / grid) reuses one
        # compiled solver (see cached_jitted_solver). The geometry is baked into
        # the closure, so the key carries it; the condition *values* are a
        # runtime argument. A traced call bypasses the cache (settings is None).
        settings = reactor_settings_key(self)
        cache_key = (
            None
            if settings is None
            else (
                "pfr",
                id(self.model),
                settings,
                self.velocity,
                self.length,
                self.n_points,
                self.n_locations,
            )
        )
        jitted = cached_jitted_solver(cache_key, self._build_jitted_solve, self.model, self.adjoint)
        with friendly_solve_errors(self.max_steps, what="plug-flow reactor solve"):
            ts, ys = jitted(C0, params, fields)
        return PFRSolution(x=ts, C=ys, model=self.model)

    def solve_sensitivity(
        self,
        C0: jnp.ndarray,
        *,
        params: jnp.ndarray,
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
        C0, conditions
            As for :meth:`solve`.
        params : jnp.ndarray
            Full parameter vector; keyword-only, matching :meth:`solve`.
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
            resolve_sens_indices,
            run_forward_sensitivity,
        )

        C0 = jnp.asarray(C0)
        params = jnp.asarray(params)
        validate_C0_params(self.model, C0, params)
        active = conditions if conditions is not None else self.conditions
        if active.n_locations != self.conditions.n_locations:
            raise ValueError(
                f"conditions override must have n_locations="
                f"{self.conditions.n_locations}, got {active.n_locations}."
            )

        free_idx = resolve_sens_indices(self.model, sens_params)
        if shared_factor is None:
            shared_factor = free_idx.shape[0] > 1
        fields = active.fields
        model = self.model
        velocity = self.velocity
        x_grid = self._x_grid
        single_loc = self.conditions.n_locations <= 1
        x_eval = jnp.linspace(0.0, self.length, self.n_points)
        atol_y = jnp.broadcast_to(jnp.asarray(self.atol, dtype=float), (model.n_species,))

        def make_f_flat(cond_arrays):
            def f_flat(x, C, p):
                cond = (
                    cond_arrays if single_loc else _interp_fields_to_scalar(x, x_grid, cond_arrays)
                )
                return model.dCdt(C, p, cond, 0) / velocity

            return f_flat

        cache_key = (
            tuple(int(i) for i in free_idx),
            bool(shared_factor),
            None if sens_rtol is None else float(sens_rtol),
        )
        # The augmented [y; S] solve resolves the sensitivity transient and is
        # step-hungrier than the primal; it honours the reactor's own max_steps
        # (the shared convention -- raise it on the reactor if the budget is hit).
        xs, y_traj, S_traj = run_forward_sensitivity(
            make_f_flat,
            C0,
            params,
            free_idx,
            fields,
            t0=0.0,
            t1=self.length,
            t_eval=x_eval,
            rtol=self.rtol,
            atol_y=atol_y,
            sens_rtol=sens_rtol,
            sens_atol=sens_atol,
            param_scale=param_scale,
            dtmax=self.dtmax,
            max_steps=self.max_steps,
            shared_factor=shared_factor,
            cache=self._sens_jit_cache,
            cache_key=cache_key,
        )
        return PFRSolution(x=xs, C=y_traj, model=model), S_traj

    def _build_jitted_solve(self):
        """Build a jit-compiled inner PFR solver. Single signature suffices."""
        model = self.model
        velocity = self.velocity
        x_grid = self._x_grid
        single_loc = self.conditions.n_locations <= 1
        length = self.length
        rtol = self.rtol
        atol = self.atol
        adjoint = self.adjoint
        dtmax = self.dtmax
        max_steps = self.max_steps
        order = self.order
        factormax = self.factormax
        solver = self.solver
        x_eval = jnp.linspace(0.0, length, self.n_points)

        @jax.jit
        def _solve(C0, params, fields):
            # Conditions at axial position x: a fixed dict for a single-location
            # reactor, else interpolated over the axial condition grid.
            def cond_fn(x):
                return fields if single_loc else _interp_fields_to_scalar(x, x_grid, fields)

            sol = solve_chemistry(
                model,
                C0,
                params,
                cond_fn=cond_fn,
                # Steady-state PFR: integrate over axial position, dC/dx =
                # (1/velocity) * dCdt.
                rate_scale=1.0 / velocity,
                saveat=diffrax.SaveAt(ts=x_eval),
                t0=0.0,
                t1=length,
                rtol=rtol,
                atol=atol,
                adjoint=adjoint,
                dtmax=dtmax,
                max_steps=max_steps,
                order=order,
                factormax=factormax,
                solver=solver,
            )
            return sol.ts, sol.ys

        return _solve
