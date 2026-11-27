"""Non-ideal flow reactors on a line of cells: axial dispersion and tanks in series.

Both reactors solve the method-of-lines system

.. math::

    \\frac{dC_i}{dt} = \\frac{F_{i-1/2} - F_{i+1/2}}{\\Delta x_i}
    + \\mathbf{S}^T r(C_i, p, \\theta_i)

on the finite-volume core of :mod:`aquakin.integrate.finite_volume`: a
transport flux ``F`` at every face, its divergence, and the canonical reaction
term ``model.dCdt`` in every cell. They differ only in the face fluxes.

- :class:`DispersedFlowReactor` -- advection with axial dispersion, the
  limited second-order scheme, a Danckwerts inlet and a zero-gradient outlet;
- :class:`TanksInSeries` -- first-order upwind advection with no dispersion
  on ``N`` equal cells, which is exactly ``N`` equal stirred tanks in series.

The inlet concentration of each species is a number or an expression of time
(the language of :func:`aquakin.core.time_expr.compile_time_expression`), read
on the model's own clock.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import diffrax
import jax
import jax.numpy as jnp

from aquakin.core.conditions import SpatialConditions
from aquakin.core.hints import did_you_mean
from aquakin.core.model import CompiledModel
from aquakin.core.time_expr import compile_time_expression, is_time_expression
from aquakin.integrate._common import (
    ExportableSolutionMixin,
    GradientCheckMixin,
    PlottableSolutionMixin,
    _HasNamedSpecies,
    _run_diffeqsolve,
    cached_jitted_solver,
    friendly_solve_errors,
    init_solver_settings,
    reactor_settings_key,
    require_increasing_t_span,
    resolve_layered_atol,
    validate_t_eval,
)
from aquakin.integrate._config import SolverSettings
from aquakin.integrate._solver import check_solve_result
from aquakin.integrate.finite_volume import (
    Grid1D,
    diffusive_flux,
    divergence,
    green_gauss_gradient,
    limited_upwind_flux,
    upwind_flux,
)
from aquakin.integrate.steady import solve_steady_state


@dataclass
class AxialFlowSolution(_HasNamedSpecies, PlottableSolutionMixin, ExportableSolutionMixin):
    """Solution returned by :class:`DispersedFlowReactor` and :class:`TanksInSeries`.

    Attributes
    ----------
    t : jnp.ndarray
        Times at which the solution was recorded, shape ``(n_t,)``, in the
        model's time unit; ``[inf]`` for a steady state.
    C : jnp.ndarray
        **Outlet** concentration, shape ``(n_t, n_species)``: the last cell
        (the last tank), which is what a tracer test measures.
        :meth:`C_named`, :meth:`to_dataframe` and :meth:`plot` read it. As for
        every reactor, this is the raw integrated state and may hold small
        transient negatives when the model clips negative states in its rates.
    profile : jnp.ndarray
        Concentration in every cell, shape ``(n_t, n_cells, n_species)``.
    x : jnp.ndarray
        Cell centres, shape ``(n_cells,)``: axial position for a
        :class:`DispersedFlowReactor`, fraction of the volume for a
        :class:`TanksInSeries`.
    model : CompiledModel
        The model that produced this solution.
    """

    t: jnp.ndarray
    C: jnp.ndarray
    profile: jnp.ndarray
    x: jnp.ndarray
    model: CompiledModel

    def profile_named(self, species: str) -> jnp.ndarray:
        """Axial profile of one species over time.

        Parameters
        ----------
        species : str
            The species name.

        Returns
        -------
        jnp.ndarray
            Shape ``(n_t, n_cells)``.

        Raises
        ------
        KeyError
            If the species is unknown.
        """
        if species not in self.model.species_index:
            suffix = did_you_mean(species, self.model.species)
            raise KeyError(f"Unknown species '{species}'. Available: {self.model.species}.{suffix}")
        return self.profile[:, :, self.model.species_index[species]]


def parse_inlet(model: CompiledModel, inlet) -> tuple[jnp.ndarray, tuple]:
    """Split an inlet specification into constants and expressions of time.

    Parameters
    ----------
    model : CompiledModel
        The model whose species the inlet names.
    inlet : array-like or mapping
        A full ``(n_species,)`` vector, or ``{species: value}`` where each value
        is a number or an expression of time ``t``; species not named enter at
        zero.

    Returns
    -------
    constants : jnp.ndarray
        Shape ``(n_species,)``: the constant inlet values, zero where an
        expression takes over.
    expressions : tuple of (int, str)
        ``(species index, expression)`` for every species given as an
        expression of time.

    Raises
    ------
    ValueError
        If the vector has the wrong shape or the mapping names an unknown
        species.
    """
    n = model.n_species
    if not isinstance(inlet, Mapping):
        constants = jnp.asarray(inlet, dtype=float)
        if constants.shape != (n,):
            raise ValueError(
                f"inlet has shape {constants.shape}; give a ({n},) vector or a "
                "{species: value} mapping."
            )
        return constants, ()
    constants = jnp.zeros(n)
    expressions = []
    for name, value in inlet.items():
        if name not in model.species_index:
            suffix = did_you_mean(name, model.species)
            raise ValueError(f"inlet names unknown species '{name}'.{suffix}")
        index = model.species_index[name]
        if is_time_expression(value):
            expressions.append((index, value))
        else:
            constants = constants.at[index].set(float(value) if isinstance(value, str) else value)
    return constants, tuple(expressions)


def _compile_inlet(model: CompiledModel, expressions) -> Callable:
    """``inlet(t, constants)``: the constants with each expression evaluated at ``t``."""
    compiled = tuple(
        (index, compile_time_expression(text, what=f"the inlet of '{model.species[index]}'"))
        for index, text in expressions
    )

    def inlet(t, constants):
        for index, fn in compiled:
            constants = constants.at[index].set(fn(t))
        return constants

    return inlet


def _require_nonnegative(value, what: str):
    """Raise unless a concrete ``value`` is finite and ``>= 0``; a traced one passes."""
    try:
        number = float(value)
    except jax.errors.ConcretizationTypeError:
        return
    if not (number >= 0 and number < float("inf")):
        raise ValueError(f"{what} must be finite and >= 0, got {number}.")


class _AxialFlowReactor(GradientCheckMixin):
    """The method-of-lines solve both axial-flow reactors share.

    A subclass sets ``self._fluxes(cells, inlet, dispersion) -> face fluxes``
    (a function of the fixed geometry only, so a compiled solve keyed on
    :meth:`_geometry_key` can be shared across instances), ``self._dispersion``
    and ``self._kind``, then calls :meth:`_setup`.
    """

    _kind: str
    _fluxes: Callable
    _dispersion: object
    _geometry_key: Callable[[], tuple]

    def _setup(self, model, conditions, grid: Grid1D, settings: SolverSettings) -> None:
        conditions.validate_required(model.conditions_required)
        n_locations = max(conditions.n_locations, 1)
        if n_locations not in (1, grid.n_cells):
            raise ValueError(
                f"conditions must have 1 location or one per cell ({grid.n_cells}), "
                f"got {n_locations}."
            )
        init_solver_settings(
            self, model, rtol=settings.rtol, integrator=settings.integrator, diff=settings.diff
        )
        self.conditions = conditions
        self.grid = grid
        self.atol = resolve_layered_atol(model, settings.atol, grid.n_cells)

    @property
    def n_cells(self) -> int:
        """Number of cells (tanks)."""
        return self.grid.n_cells

    def _initial_profile(self, C0) -> jnp.ndarray:
        C0 = jnp.asarray(C0, dtype=float)
        shape = (self.n_cells, self.model.n_species)
        if C0.shape == shape[1:]:
            return jnp.broadcast_to(C0, shape)
        if C0.shape == shape:
            return C0
        raise ValueError(
            f"C0 has shape {C0.shape}; give one value per species {shape[1:]} or a profile {shape}."
        )

    def _inputs(self, params, conditions):
        model = self.model
        params = model.default_parameters() if params is None else jnp.asarray(params)
        if params.shape != (model.n_params,):
            raise ValueError(f"params has shape {params.shape}, expected ({model.n_params},)")
        active = conditions if conditions is not None else self.conditions
        if active.n_locations != self.conditions.n_locations:
            raise ValueError(
                f"conditions override must have n_locations={self.conditions.n_locations}, "
                f"got {active.n_locations}."
            )
        return params, active.fields

    def _make_rhs(self, fields, inlet: Callable, stoich=None) -> Callable:
        """``rhs(t, cells, (params, dispersion, inlet_constants))``: transport plus reaction.

        ``stoich`` is the stoichiometry hoisted from the parameters of a solve;
        ``None`` evaluates it from the ``params`` in ``args`` at every call, so a
        derivative with respect to them (the steady state's implicit-function
        gradient) reaches the parameter-dependent coefficients.
        """
        model = self.model
        fluxes = self._fluxes
        widths = self.grid.widths
        n = self.n_cells
        location = jnp.arange(n) if self.conditions.n_locations == n > 1 else jnp.zeros(n, int)

        def reaction(c, p, loc):
            return model.dCdt(c, p, fields, loc, stoich=stoich)

        def rhs(t, cells, args):
            p, dispersion, constants = args
            transport = divergence(fluxes(cells, inlet(t, constants), dispersion), widths)
            return transport + jax.vmap(reaction, in_axes=(0, None, 0))(cells, p, location)

        return rhs

    def solve(
        self,
        C0,
        t_span: tuple[float, float],
        t_eval: jnp.ndarray | None = None,
        *,
        inlet,
        params: jnp.ndarray | None = None,
        conditions: SpatialConditions | None = None,
    ) -> AxialFlowSolution:
        """Integrate the reactor over a time span.

        Parameters
        ----------
        C0 : array-like
            Initial concentration: one value per species, shape
            ``(n_species,)``, filling every cell, or a profile of shape
            ``(n_cells, n_species)``.
        t_span : tuple of float
            ``(t_start, t_end)``, in the model's time unit.
        t_eval : jnp.ndarray, optional
            Times at which to record the solution; ``None`` records the end
            point only.
        inlet : array-like or mapping
            Inlet concentration: a ``(n_species,)`` vector, or ``{species:
            value}`` where each value is a number or an expression of time
            ``t`` on the model's clock (``"step(t - 10)"``, ``"5*exp(-t/30)"``);
            species not named enter at zero.
        params : jnp.ndarray, optional
            Rate constant vector, shape ``(n_params,)``. Defaults to
            ``model.default_parameters()``.
        conditions : SpatialConditions, optional
            Override the reactor's conditions for this call, with the same
            number of locations.

        Returns
        -------
        AxialFlowSolution

        Raises
        ------
        ValueError
            On a wrongly shaped ``C0``, ``inlet`` or ``params``, an unknown
            inlet species, or an invalid time span.
        aquakin.core.time_expr.TimeExpressionError
            If an inlet expression cannot be read.
        """
        y0 = self._initial_profile(C0)
        params, fields = self._inputs(params, conditions)
        constants, expressions = parse_inlet(self.model, inlet)
        inlet_fn = _compile_inlet(self.model, expressions)
        t0, t1 = require_increasing_t_span(t_span)
        t_eval_arr = None if t_eval is None else jnp.asarray(t_eval)
        if t_eval_arr is not None:
            validate_t_eval(t_eval_arr, t0, t1)

        settings = reactor_settings_key(self)
        cache_key = (
            None
            if settings is None
            else (
                self._kind,
                id(self.model),
                settings,
                self._geometry_key(),
                self.conditions.n_locations,
                expressions,
                (t0, t1, None if t_eval_arr is None else t_eval_arr.shape),
            )
        )
        jitted = cached_jitted_solver(
            cache_key,
            lambda: self._build_jitted_solve(inlet_fn, t0, t1, t_eval_arr is not None),
            self.model,
            self.adjoint,
        )
        what = f"{type(self).__name__} solve"
        with friendly_solve_errors(what=what):
            ts, ys, result = jitted(y0, params, self._dispersion, constants, fields, t_eval_arr)
            ts, ys = check_solve_result(result, (ts, ys), max_steps=self.max_steps, what=what)
        return AxialFlowSolution(
            t=ts, C=ys[:, -1, :], profile=ys, x=jnp.asarray(self.grid.centres), model=self.model
        )

    def _build_jitted_solve(self, inlet_fn, t0, t1, has_t_eval):
        """Build the compiled solve; geometry, inlet expressions and span are baked in."""
        make_rhs = self._make_rhs
        model = self.model
        kw = dict(
            t0=t0,
            t1=t1,
            rtol=self.rtol,
            atol=self.atol,
            adjoint=self.adjoint,
            dtmax=self.dtmax,
            max_steps=self.max_steps,
            order=self.order,
            factormax=self.factormax,
            solver=self.solver,
        )

        @jax.jit
        def _solve(y0, params, dispersion, constants, fields, t_eval):
            saveat = diffrax.SaveAt(ts=t_eval) if has_t_eval else diffrax.SaveAt(t1=True)
            sol = _run_diffeqsolve(
                make_rhs(fields, inlet_fn, stoich=model.compute_stoich(params)),
                y0=y0,
                args=(params, dispersion, constants),
                saveat=saveat,
                **kw,
            )
            return sol.ts, sol.ys, sol.result

        return _solve

    def steady_state(
        self,
        inlet,
        *,
        params: jnp.ndarray | None = None,
        conditions: SpatialConditions | None = None,
        guess=None,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> AxialFlowSolution:
        """Solve for the steady profile under a constant inlet.

        The root of the right-hand side is found by pseudo-transient
        continuation (:func:`aquakin.integrate.steady.solve_steady_state`), and
        is differentiable with respect to ``params``, the dispersion and the
        inlet through the implicit function theorem.

        Parameters
        ----------
        inlet : array-like or mapping
            As for :meth:`solve`, but every value must be a constant.
        params : jnp.ndarray, optional
            Rate constant vector. Defaults to ``model.default_parameters()``.
        conditions : SpatialConditions, optional
            Override the reactor's conditions for this call.
        guess : array-like, optional
            Starting profile, as ``C0`` of :meth:`solve`. Defaults to the inlet
            in every cell.
        tol : float, optional
            Convergence tolerance on the scaled residual
            ``max_i |f_i| / max(|y_i|, 1)``.
        max_iter : int, optional
            Iteration cap.

        Returns
        -------
        AxialFlowSolution
            With ``t = [inf]`` and one profile.

        Raises
        ------
        ValueError
            If an inlet value is an expression of time, or on the input
            errors of :meth:`solve`.
        RuntimeError
            If the iteration does not converge within ``max_iter``.
        """
        params, fields = self._inputs(params, conditions)
        constants, expressions = parse_inlet(self.model, inlet)
        if expressions:
            names = [self.model.species[index] for index, _ in expressions]
            raise ValueError(
                f"steady_state needs a constant inlet; {names} are given as expressions of time."
            )
        y0 = self._initial_profile(constants if guess is None else guess)
        cache_key = (
            f"{self._kind}_steady",
            id(self.model),
            self._geometry_key(),
            self.conditions.n_locations,
            float(tol),
            int(max_iter),
        )
        steady = cached_jitted_solver(
            cache_key, lambda: self._build_steady_solve(tol, max_iter), self.model
        )
        state, residual, converged = steady(y0, params, self._dispersion, constants, fields)
        try:
            converged = bool(converged)
        except jax.errors.ConcretizationTypeError:
            converged = True  # traced under an outer jit: the flag cannot be read here
        if not converged:
            raise RuntimeError(
                f"{type(self).__name__}.steady_state did not converge in {max_iter} "
                f"iterations (scaled residual {float(residual):.3g} > tol={tol}). "
                "Raise max_iter, or start from a guess= nearer the steady profile."
            )
        y = state.reshape(y0.shape)
        return AxialFlowSolution(
            t=jnp.asarray([jnp.inf]),
            C=y[-1][None, :],
            profile=y[None],
            x=jnp.asarray(self.grid.centres),
            model=self.model,
        )

    def _build_steady_solve(self, tol, max_iter):
        """Build the compiled steady-state solve; the geometry is baked in."""
        make_rhs = self._make_rhs

        @jax.jit
        def _steady(y0, params, dispersion, constants, fields):
            rhs = make_rhs(fields, lambda t, c: c)

            def rhs_flat(y, args):
                return rhs(0.0, y.reshape(y0.shape), args).reshape(-1)

            result = solve_steady_state(
                rhs_flat,
                (params, dispersion, constants),
                jax.lax.stop_gradient(y0).reshape(-1),
                tol=tol,
                max_iter=max_iter,
                scale_floor=1.0,
                nonneg=True,
            )
            return result.state, result.residual, result.converged

        return _steady


def _dispersed_flow_fluxes(grid: Grid1D, velocity: float) -> Callable:
    """Face fluxes of advection with axial dispersion, Danckwerts in, zero gradient out."""
    face_velocity = jnp.full(grid.n_cells + 1, velocity)
    # Half the first cell: the distance from the inlet face to the first centre.
    half = float(grid.face_spacing[0])

    def fluxes(cells, inlet, dispersion):
        # Danckwerts: the total flux through the inlet face is u * C_in, so the
        # face value c_b satisfies u C_in = u c_b - D (C_0 - c_b) / half.
        conductance = dispersion / half
        inlet_face = (velocity * inlet + conductance * cells[0]) / (velocity + conductance)
        outlet_face = cells[-1]
        gradient = green_gauss_gradient(cells, grid, inlet_face, outlet_face)
        advective = limited_upwind_flux(
            cells, face_velocity, grid, inlet_face, outlet_face, gradient
        )
        nodes = jnp.concatenate([inlet_face[None, :], cells, outlet_face[None, :]], axis=0)
        diffusivity = jnp.broadcast_to(dispersion, (cells.shape[1],))
        return advective + diffusive_flux(nodes, diffusivity, grid.face_spacing)

    return fluxes


class DispersedFlowReactor(_AxialFlowReactor):
    """One-dimensional advection--dispersion--reaction reactor.

    Solves ``dC/dt = -u dC/dx + D d2C/dx2 + R(C)`` over ``0 <= x <= length``
    with a constant velocity ``u`` and axial dispersion coefficient ``D``, the
    standard model of a non-ideal contactor, a packed or soil column, or a
    river reach. ``D -> 0`` recovers the :class:`PlugFlowReactor`; ``D -> inf``
    a single stirred tank.

    **Scheme.** Cell-centred finite volume on ``n_cells`` equal cells. The
    advective face value is reconstructed from the upwind cell with its
    Green--Gauss gradient, limited by the smooth van Albada limiter
    (:func:`~aquakin.integrate.finite_volume.limited_upwind_flux`): second
    order in space where the profile is smooth, first order at extrema and
    steep fronts, without oscillation, and differentiable throughout. The
    dispersive flux is the central difference ``D (C_i - C_(i+1)) / dx``
    (second order). The inlet is Danckwerts -- the total flux through it is
    ``u C_in`` -- and the outlet is zero-gradient. Time integration is the
    adaptive implicit solver every reactor uses.

    Parameters
    ----------
    model : CompiledModel
        Compiled reaction model. Every species disperses with the same ``D``.
    conditions : SpatialConditions
        One location (uniform), or one per cell.
    length : float
        Reactor length.
    velocity : float
        Mean (interstitial) velocity, in length per the model's time unit.
    n_cells : int, optional
        Number of cells (default 50).
    dispersion : float, optional
        Axial dispersion coefficient ``D``, in length² per the model's time
        unit.
    dispersion_number : float, optional
        ``D / (u L)``, the reciprocal of the Péclet number. Give exactly one of
        ``dispersion`` and ``dispersion_number``. Either may be a traced value,
        so a gradient can be taken with respect to it.
    settings : SolverSettings, optional
        Tolerances and solver configuration.

    Raises
    ------
    ValueError
        If both or neither of ``dispersion`` and ``dispersion_number`` are
        given, either is negative, the geometry is not positive, or the
        conditions have neither one location nor one per cell.
    """

    _kind = "dispersed_flow"

    def __init__(
        self,
        model: CompiledModel,
        conditions: SpatialConditions,
        length: float,
        velocity: float,
        n_cells: int = 50,
        *,
        dispersion=None,
        dispersion_number=None,
        settings: SolverSettings = SolverSettings(),
    ) -> None:
        if (dispersion is None) == (dispersion_number is None):
            raise ValueError(
                "Give exactly one of dispersion= (the coefficient D) and "
                "dispersion_number= (D / (u L))."
            )
        if not velocity > 0:
            raise ValueError(f"velocity must be positive, got {velocity}.")
        grid = Grid1D.uniform(length, n_cells)
        self.length = float(length)
        self.velocity = float(velocity)
        if dispersion is None:
            _require_nonnegative(dispersion_number, "dispersion_number")
            dispersion = dispersion_number * self.velocity * self.length
        else:
            _require_nonnegative(dispersion, "dispersion")
        self._dispersion = dispersion
        self._fluxes = _dispersed_flow_fluxes(grid, self.velocity)
        self._setup(model, conditions, grid, settings)

    @property
    def dispersion(self):
        """The axial dispersion coefficient ``D``."""
        return self._dispersion

    @property
    def dispersion_number(self):
        """``D / (u L)``, the reciprocal of the Péclet number."""
        return self._dispersion / (self.velocity * self.length)

    def _geometry_key(self) -> tuple:
        return (self.length, self.velocity, self.n_cells)


def _tanks_fluxes(n_tanks: int, velocity: float) -> Callable:
    """First-order upwind face fluxes with no dispersion: the tanks-in-series exchange."""
    face_velocity = jnp.full(n_tanks + 1, velocity)

    def fluxes(cells, inlet, dispersion):
        return upwind_flux(cells, face_velocity, inlet, cells[-1])

    return fluxes


class TanksInSeries(_AxialFlowReactor):
    """``N`` equal stirred tanks in series, in one state vector.

    Each tank of volume ``V/N`` takes the outflow of the one before it:
    ``dC_i/dt = (N / tau) (C_(i-1) - C_i) + R(C_i)``, with ``C_0`` the inlet
    and ``tau`` the residence time of the whole train. It is the finite-volume
    core with first-order upwind advection and no dispersion on ``N`` equal
    cells, which is exactly this system. ``N = 1`` is a single CSTR; ``N ->
    inf`` approaches plug flow.

    Parameters
    ----------
    model : CompiledModel
        Compiled reaction model.
    conditions : SpatialConditions
        One location (uniform), or one per tank.
    n_tanks : int
        Number of tanks, at least 1.
    residence_time : float
        Hydraulic residence time of the whole train, ``V / Q``, in the
        model's time unit.
    settings : SolverSettings, optional
        Tolerances and solver configuration.

    Raises
    ------
    ValueError
        If ``n_tanks`` is not a positive integer or ``residence_time`` is not
        positive.
    """

    _kind = "tanks_in_series"
    _dispersion = 0.0

    def __init__(
        self,
        model: CompiledModel,
        conditions: SpatialConditions,
        n_tanks: int,
        residence_time: float,
        *,
        settings: SolverSettings = SolverSettings(),
    ) -> None:
        if isinstance(n_tanks, bool) or int(n_tanks) != n_tanks or n_tanks < 1:
            raise ValueError(f"n_tanks must be a positive integer, got {n_tanks}.")
        if not residence_time > 0:
            raise ValueError(f"residence_time must be positive, got {residence_time}.")
        self.residence_time = float(residence_time)
        # Unit length, so a cell's width is its tank's share of the volume.
        grid = Grid1D.uniform(1.0, int(n_tanks))
        self._fluxes = _tanks_fluxes(grid.n_cells, 1.0 / self.residence_time)
        self._setup(model, conditions, grid, settings)

    @property
    def n_tanks(self) -> int:
        """Number of tanks."""
        return self.n_cells

    def _geometry_key(self) -> tuple:
        return (self.residence_time, self.n_cells)
