"""Batch (0-D) reactor: integrate chemistry at a single spatial location."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

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
    cached_jitted_solver,
    friendly_solve_errors,
    init_solver_settings,
    make_chemistry_rhs,
    reactor_settings_key,
    resolve_state_atol,
    solve_chemistry,
    to_native_time,
    validate_C0_params,
    validate_t_eval,
)
from aquakin.integrate.events import Event, solve_with_events


@dataclass
class BatchSolution(_HasNamedSpecies, PlottableSolutionMixin, ExportableSolutionMixin):
    """
    Solution returned by :meth:`BatchReactor.solve`.

    Attributes
    ----------
    t : jnp.ndarray
        Times at which the solution was recorded, shape ``(n_t,)``.
    C : jnp.ndarray
        Concentration trajectory, shape ``(n_t, n_species)``. This is the raw
        integrated state. If the model sets ``clip_negative_states`` (on by
        default for ASM1), individual entries may be **small transient
        negatives**: the ``max(C, 0)`` clamp is applied only when evaluating the
        reaction rates (and state-derived conditions such as pH), not to the
        saved state, so the rates were computed on the clamped values while the
        trajectory keeps the raw ones. These negatives are a normal numerical
        transient of a stiff solve, not a solver or model error; clip them
        yourself (``jnp.maximum(sol.C, 0.0)``) for display if needed.
    model : CompiledModel
        The model that produced this solution. Retained so that the
        inherited :meth:`C_named` can look up species by name.
    events_log : list of (float, str), optional
        When the solve used ``events=``, the fired events in order as
        ``(time, name)`` -- the audit trail of switch times. ``None`` for a plain
        solve.
    """

    t: jnp.ndarray
    C: jnp.ndarray
    model: CompiledModel
    events_log: Optional[list] = None


class BatchReactor(GradientCheckMixin):
    """
    Stateless 0-D (batch) reactor.

    Parameters
    ----------
    model : CompiledModel
        Compiled reaction model.
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
    integrator : IntegratorConfig, optional
        Integrator / step-size configuration (ESDIRK ``order``, ``factormax``,
        ``dtmax``, ``max_steps``, an explicit ``solver``). The default flips the
        reactor to the fast stack (Kvaerno3 + capped step growth). ``dtmax``
        is always in the model's **native** time unit (the unit of its rate
        constants -- seconds for ozone/UV, days for ASM/ADM/WATS), independent of
        any ``time_unit=`` passed to :meth:`solve`.
    diff : DifferentiationConfig, optional
        Autodiff configuration. ``mode="reverse"`` (default) differentiates the
        solve with ``RecursiveCheckpointAdjoint`` (reactors have no stable
        reverse adjoint -- that is plant-only -- so ``method`` is ignored for
        reverse mode). ``mode="forward", method="through_solve"`` builds a
        forward-capable ``DirectAdjoint`` so ``jax.jvp`` / ``jax.jacfwd`` flow
        through the solve; ``mode="forward", method="stable"`` is the augmented
        ``solve_sensitivity`` path (cap-free).

    Notes
    -----
    **Differentiating a stiff solve.** A reverse-mode gradient (``jax.grad`` /
    ``jax.jacrev``) taken directly through ``solve`` on a stiff model
    (ASM / ADM / WATS) returns **silent** ``NaN`` / ``Inf`` when ``dtmax`` is
    uncapped -- the backward accumulation overflows (see ``dtmax`` above). No
    exception is raised, so the non-finite gradient flows into an optimizer as
    garbage and the fit never converges with no indication why. :func:`aquakin.calibrate`
    and :func:`aquakin.sensitivity` guard this (``check_finite=True``), but a
    hand-rolled loss + optimizer through ``solve`` is exposed. The remedies, in
    order of convenience:

    - build the reactor with a ``dtmax`` cap (the simplest fix);
    - differentiate in **forward mode** (``jax.jacfwd`` with
      ``adjoint=aquakin.forward_adjoint()``), finite at any step;
    - use :func:`aquakin.calibrate` (``gradient="stable_adjoint"``, cap-free) or
      :func:`aquakin.sensitivity`, which handle it internally;
    - guard your own gradient with :meth:`check_gradient_finite`
      (``g = reactor.check_gradient_finite(jax.grad(loss)(params))``), which
      raises an actionable error instead of returning silent ``NaN``.
    """

    def __init__(
        self,
        model: CompiledModel,
        conditions: SpatialConditions,
        *,
        rtol: float = 1e-6,
        atol=None,
        integrator: IntegratorConfig = IntegratorConfig(),
        diff: DifferentiationConfig = DifferentiationConfig(),
    ) -> None:
        conditions.validate_required(model.conditions_required)
        init_solver_settings(self, model, rtol=rtol, integrator=integrator, diff=diff)
        self.conditions = conditions
        self.atol = resolve_state_atol(model, atol)
        # Cache jit-compiled inner solve keyed on (t0, t1, t_eval_shape).
        # First call with a new signature pays the trace cost; subsequent
        # calls reuse the compiled graph.
        self._jit_cache: dict = {}
        # Separate cache for the forward-sensitivity solves.
        self._sens_jit_cache: dict = {}

    def solve(
        self,
        C0: jnp.ndarray,
        t_span: tuple[float, float] = None,
        t_eval: Optional[jnp.ndarray] = None,
        *,
        params: Optional[jnp.ndarray] = None,
        conditions: Optional[SpatialConditions] = None,
        time_unit: Optional[str] = None,
        events: Optional[Sequence[Event]] = None,
    ) -> BatchSolution:
        """
        Integrate the reaction model over a time span.

        Parameters
        ----------
        C0 : jnp.ndarray
            Initial concentration vector, shape ``(n_species,)``.
        t_span : tuple of float
            ``(t_start, t_end)`` integration interval, in the model's time unit
            unless ``time_unit`` is given. The required second positional argument
            (``solve(C0, (0.0, 600.0))``).
        t_eval : jnp.ndarray, optional
            Time points at which to record solution. If ``None`` the solver
            returns endpoints only.
        params : jnp.ndarray, optional, keyword-only
            Rate constant vector, shape ``(n_params,)``. Defaults to
            ``model.default_parameters()`` -- pass it (as a keyword) only to
            override rate constants (e.g. a what-if run; see
            ``model.parameter_values``). Keyword-only so a positional ``t_span``
            tuple can never land in it.
        conditions : SpatialConditions, optional
            Override the conditions stored on the reactor for this call. Used
            by :func:`aquakin.sensitivity` to differentiate through condition
            fields without mutating the reactor.
        time_unit : str, optional
            The time unit ``t_span`` / ``t_eval`` are expressed in (``"s"``,
            ``"min"``, ``"h"``, ``"d"``). aquakin has no global time unit -- the
            native unit is set by the model's rate constants
            (``model.time_unit``: seconds for ozone/UV, days for ASM/ADM/WATS).
            Pass ``time_unit`` to work in a different unit: the input times are
            converted into the native unit for the solve (rate constants
            unchanged) and the returned ``solution.t`` is reported back in
            ``time_unit`` (with ``solution.time_unit`` set to it). Default
            ``None`` uses the model's native unit. Raises if the model's own
            time unit is undeclared (``model.time_unit is None``), since there
            is then no native unit to convert to.
        events : sequence of Event, optional
            Located discontinuities (on/off switches, SBR phases, level limits)
            applied during the solve. Each :class:`~aquakin.Event` fires at a
            known time (``at_times=``) or when a state ``cond_fn`` crosses zero,
            and may reset the state (``apply=``) or terminate the solve
            (``terminal=``). The solve is split into segments at the firings; the
            returned ``solution.events_log`` records them. Time-only events keep
            ``jax.grad`` finite (static segment boundaries); a state event makes
            the solve a forward simulation (the firing count is data-dependent).
            See :func:`aquakin.solve_with_events`.

        Returns
        -------
        BatchSolution
        """
        C0 = jnp.asarray(C0)
        params = self.model.default_parameters() if params is None else jnp.asarray(params)
        validate_C0_params(self.model, C0, params)
        if t_span is None:
            raise ValueError("t_span=(t_start, t_end) is required.")

        t_span, t_eval, _time_factor = to_native_time(
            self.model.time_unit, time_unit, t_span, t_eval
        )
        # ``time_unit`` rescales t_span/t_eval into native time, but ``dtmax`` is
        # always native (it caps the integrator's own step). A user who set both
        # in the requested unit would get a cap off by the unit ratio, so warn
        # that dtmax stays native and must be scaled by hand.
        if _time_factor != 1.0 and self.dtmax is not None:
            warnings.warn(
                f"dtmax={self.dtmax} is in the model's native time unit "
                f"('{self.model.time_unit}'), but solve(time_unit='{time_unit}') "
                f"rescales t_span/t_eval; dtmax is NOT rescaled. Express dtmax in "
                f"native units (multiply your intended cap by {_time_factor:g}).",
                stacklevel=2,
            )

        t0, t1 = float(t_span[0]), float(t_span[1])
        if not (t1 > t0):
            raise ValueError(f"t_span end must exceed start; got ({t0}, {t1}).")
        active_conditions = conditions if conditions is not None else self.conditions
        condition_arrays = active_conditions.fields

        if events is not None:
            return self._solve_with_events(
                C0, params, condition_arrays, t0, t1, t_eval, events, _time_factor
            )

        if t_eval is None:
            t_eval_arr = None
            sig = (t0, t1, None)
        else:
            t_eval_arr = jnp.asarray(t_eval)
            self._validate_t_eval(t_eval_arr, t0, t1)
            sig = (t0, t1, tuple(t_eval_arr.shape))

        # Shared across reactor instances: the same model + settings + call
        # signature reuses one compiled solver (see cached_jitted_solver). A
        # traced call (solve inside an outer jit/grad) yields a None settings key
        # and bypasses the cache -- it cannot benefit from it anyway.
        settings = reactor_settings_key(self)
        cache_key = None if settings is None else ("batch", id(self.model), sig, settings)
        jitted = cached_jitted_solver(
            cache_key,
            lambda: self._build_jitted_solve(t0, t1, t_eval_arr is not None),
            self.model,
            self.adjoint,
        )

        with friendly_solve_errors(self.max_steps, what="batch reactor solve"):
            if t_eval_arr is None:
                ts, ys = jitted(C0, params, condition_arrays)
            else:
                ts, ys = jitted(C0, params, condition_arrays, t_eval_arr)
        if _time_factor != 1.0:
            ts = ts / _time_factor  # native -> requested unit
        sol = BatchSolution(t=ts, C=ys, model=self.model)
        if time_unit is not None:
            sol._requested_time_unit = time_unit
        return sol

    def _solve_with_events(self, C0, params, condition_arrays, t0, t1, t_eval, events, time_factor):
        """Run the event-driven segmented solve (the ``events=`` path).

        Builds the same constant-condition RHS the plain batch solve uses --
        through the shared :func:`make_chemistry_rhs` factory, so the event path
        and ``solve_chemistry`` cannot drift -- and hands it to
        :func:`solve_with_events`, which locates the events and applies their
        resets between segments. Not routed through the jit cache: the driver is
        an eager Python loop over segments (a state event's count is
        data-dependent), and time-only events still differentiate because each
        segment is a plain differentiable sub-solve.
        """
        rhs = make_chemistry_rhs(self.model, params, cond_fn=lambda t: condition_arrays)

        t_eval_arr = None if t_eval is None else jnp.asarray(t_eval)
        if t_eval_arr is not None:
            self._validate_t_eval(t_eval_arr, t0, t1)
        with friendly_solve_errors(self.max_steps, what="batch reactor solve"):
            res = solve_with_events(
                rhs,
                C0,
                params,
                t0=t0,
                t1=t1,
                t_eval=t_eval_arr,
                events=events,
                rtol=self.rtol,
                atol=self.atol,
                dtmax=self.dtmax,
                adjoint=self.adjoint,
                max_steps=self.max_steps,
                order=self.order,
                factormax=self.factormax,
                solver=self.solver,
            )
        ts = res.ts / time_factor if time_factor != 1.0 else res.ts
        return BatchSolution(t=ts, C=res.ys, model=self.model, events_log=res.log)

    def solve_sensitivity(
        self,
        C0: jnp.ndarray,
        t_span: tuple[float, float],
        t_eval: Optional[jnp.ndarray] = None,
        *,
        params: jnp.ndarray,
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
        C0, t_span, t_eval, conditions
            As for :meth:`solve`.
        params : jnp.ndarray
            Full parameter vector; keyword-only, matching :meth:`solve`.
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
            resolve_sens_indices,
            run_forward_sensitivity,
        )

        C0 = jnp.asarray(C0)
        params = jnp.asarray(params)
        validate_C0_params(self.model, C0, params)
        t0, t1 = float(t_span[0]), float(t_span[1])
        if not (t1 > t0):
            raise ValueError(f"t_span end must exceed start; got ({t0}, {t1}).")

        free_idx = resolve_sens_indices(self.model, sens_params)
        if shared_factor is None:
            shared_factor = free_idx.shape[0] > 1
        active = conditions if conditions is not None else self.conditions
        cond = active.fields
        model = self.model
        atol_y = jnp.broadcast_to(jnp.asarray(self.atol, dtype=float), (model.n_species,))
        t_eval_arr = None if t_eval is None else jnp.asarray(t_eval)
        if t_eval_arr is not None:
            self._validate_t_eval(t_eval_arr, t0, t1)

        def make_f_flat(condition_arrays):
            return lambda t, y, p: model.dCdt(y, p, condition_arrays, 0)

        cache_key = (
            t0,
            t1,
            None if t_eval_arr is None else tuple(t_eval_arr.shape),
            tuple(int(i) for i in free_idx),
            bool(shared_factor),
            None if sens_rtol is None else float(sens_rtol),
        )
        ts, y_traj, S_traj = run_forward_sensitivity(
            make_f_flat,
            C0,
            params,
            free_idx,
            cond,
            t0=t0,
            t1=t1,
            t_eval=t_eval_arr,
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
        return BatchSolution(t=ts, C=y_traj, model=model), S_traj

    _validate_t_eval = staticmethod(validate_t_eval)

    def _build_jitted_solve(self, t0: float, t1: float, has_t_eval: bool):
        """Build a jit-compiled inner solver for a specific call signature."""
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

        if has_t_eval:

            @jax.jit
            def _solve(C0, params, condition_arrays, t_eval):
                sol = solve_chemistry(
                    model,
                    C0,
                    params,
                    cond_fn=lambda t: condition_arrays,
                    saveat=diffrax.SaveAt(ts=t_eval),
                    **kw,
                )
                return sol.ts, sol.ys

            return _solve

        @jax.jit
        def _solve(C0, params, condition_arrays):
            sol = solve_chemistry(
                model,
                C0,
                params,
                cond_fn=lambda t: condition_arrays,
                saveat=diffrax.SaveAt(t1=True),
                **kw,
            )
            return sol.ts, sol.ys

        return _solve
