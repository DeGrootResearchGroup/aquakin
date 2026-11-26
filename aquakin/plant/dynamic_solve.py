"""The dynamic solve of a :class:`~aquakin.plant.plant.Plant`.

:func:`solve` integrates the assembled plant over a time span and routes the
request to one of four paths: the cached forward solve (``jax_adjoint``), the
lean ``forward_fast`` loop, the cap-free discrete adjoint (``stable_adjoint``)
and the segmented located-event solve. The three jit builders that compile
those paths, the argument validation and the one-time diagnostics they share
live here too.

Every function takes the plant as its first argument and is bound onto
:class:`Plant` as a method (see ``plant.py``), so ``plant.solve(...)`` is
unchanged. The first argument keeps the name ``self`` because these are the
plant's methods: ``inspect.signature(Plant.solve)`` is part of the public
contract, and it lists that argument.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import diffrax
import jax
import jax.numpy as jnp

from aquakin.integrate._common import (
    DifferentiationConfig,
    IntegratorConfig,
    concrete_settings_key,
    friendly_solve_errors,
    to_native_time,
)
from aquakin.integrate._config import FORWARD_THROUGH_SOLVE_DIFF, STABLE_REVERSE_DIFF
from aquakin.integrate._solver import check_solve_result
from aquakin.integrate.discrete_adjoint import esdirk_adjoint_solve
from aquakin.integrate.events import (
    Event,
    SegmentSettings,
    build_segment_solver,
    events_to_native_time,
    root_events_of,
    run_segmented,
    segment_signature,
)
from aquakin.integrate.seam import _coerce_atol, _run_diffeqsolve
from aquakin.plant.influent import warn_if_outside_records
from aquakin.plant.results import (
    PlantSolution,
)
from aquakin.plant.units import (
    CycleEventSource,
    InletFlowCheckedUnit,
)


def _check_inlet_flows(self, t0: float, t1: float) -> None:
    """Let every :class:`~aquakin.plant.units.InletFlowCheckedUnit` check its
    inlets over the solve span ``(t0, t1)`` (plant clock) before integrating.

    Each input port is given the influent feeding it (already on the plant's
    clock) or ``None`` when another unit feeds it. Raises what the unit raises.
    """
    for name, unit in self.units.items():
        if not isinstance(unit, InletFlowCheckedUnit):
            continue
        inlets = {}
        for port in unit.input_ports:
            source = next(
                (c for c in self.connections if c.to_unit == name and c.to_port == port), None
            )
            fed_by_influent = source is not None and source.from_unit is None
            inlets[port] = self.influents[source.from_port] if fed_by_influent else None
        unit.check_inlet_flows(inlets, t0, t1)


def _concrete_teval_key(t_eval):
    """Hashable cache key for a concrete ``t_eval`` (or ``None``).

    Returns ``(key, is_concrete)``. A traced ``t_eval`` (the solve running under
    an outer trace) cannot be materialised into a key, so ``is_concrete`` is
    ``False`` and the caller bypasses the compiled-solve cache.
    """
    if t_eval is None:
        return None, True
    try:
        return tuple(float(x) for x in jnp.asarray(t_eval)), True
    except (jax.errors.ConcretizationTypeError, TypeError):
        return None, False


def _auto_colored_backward(n_states, min_states):
    """The ``colored_jacobian="auto"`` choice for the stable-adjoint backward.

    The one size gate behind both :meth:`Plant.colored_jacobian_decision` and the
    stable-adjoint solve, so the reported choice is the one the solve makes.

    Parameters
    ----------
    n_states : int
        The plant's flat state length.
    min_states : int
        The gate, :attr:`Plant._COLORED_BACKWARD_MIN_STATES` (read off the
        instance, so a per-plant override applies).

    Returns
    -------
    str
        ``"colored"`` when ``n_states >= min_states``, else ``"dense"``.
    """
    return "colored" if n_states >= min_states else "dense"


def solve(
    self,
    t_span: tuple[float, float],
    t_eval: jnp.ndarray | None = None,
    params: jnp.ndarray | None = None,
    *,
    rtol: float = 1e-6,
    atol: float | jnp.ndarray | None = None,
    y0: jnp.ndarray | None = None,
    integrator: IntegratorConfig = IntegratorConfig(),
    diff: DifferentiationConfig = DifferentiationConfig(),
    time_unit: str | None = None,
    event: diffrax.Event | None = None,
    events: Sequence[Event] | None = None,
    progress_meter: diffrax.AbstractProgressMeter | None = None,
    forward_fast: bool = False,
) -> PlantSolution:
    """Integrate the plant over ``t_span``.

    Parameters
    ----------
    t_span : (float, float)
        ``(t_start, t_end)`` in the plant's time unit (``plant.time_unit``,
        typically days for BSM-family) unless ``time_unit`` is given.
    t_eval : jnp.ndarray, optional
        Save times. If ``None`` only the endpoint is saved.
    params : jnp.ndarray, optional
        Flat plant parameter vector. Defaults to :meth:`default_parameters`.
    rtol, atol : float or array, optional
        Solver tolerances. ``atol=None`` auto-scales a per-component noise
        floor off the operating magnitudes.
    y0 : jnp.ndarray, optional
        Warm-start initial state (e.g. a previously-computed steady state),
        shape ``(total_state_size,)``. Defaults to the per-unit initial states.
    integrator : IntegratorConfig, optional
        Integrator / step configuration. ``order`` selects Kvaerno3 (default,
        fast) or Kvaerno5; ``factormax`` caps the PID step-growth (default 3);
        ``dtmax`` caps the step (set it only for a reverse gradient *through*
        the solve, ``diff.method='through_solve'``); ``max_steps`` is the step
        budget; ``solver`` is an explicit override (honoured verbatim);
        ``colored_jacobian`` ({"auto", True, False}) selects sparse colored-AD
        materialisation of the per-step Jacobian. ``"auto"`` (default) governs
        the ``mode='reverse', method='stable'`` backward decision -- it enables
        the colored ``df/dy`` build by plant size (BSM1 and BSM2 yes, a plant
        of a few tanks no), and leaves the forward solve dense; ``True`` forces
        coloring on both paths; ``False`` disables it. Reported by
        :meth:`colored_jacobian_decision`.
    diff : DifferentiationConfig, optional
        How a gradient / sensitivity flows through the solve.
        ``mode='reverse', method='stable'`` (the default) routes a plain
        concrete forward solve to the fast cached path and a reverse-mode
        differentiated solve to the cap-free hand-written discrete adjoint
        (:func:`~aquakin.esdirk_adjoint_solve`) -- finite for a stiff plant
        with no ``dtmax`` to tune. ``mode='reverse', method='through_solve'``
        differentiates *through* the diffrax solve (``RecursiveCheckpointAdjoint``;
        needs a ``dtmax`` cap for a stiff plant). ``mode='forward',
        method='through_solve'`` builds a forward-capable adjoint so
        ``jax.jvp`` / ``jacfwd`` flow through the solve. ``mode='forward',
        method='stable'`` is the augmented variational solve -- not supported
        here; call :meth:`solve_sensitivity` / :meth:`dynamic_sensitivity`.
        ``check_finite`` raises on a non-finite gradient; ``adjoint_max_steps``
        bounds the discrete-adjoint backward-scan buffer (it must exceed the
        forward step count); ``adjoint_low_memory`` recomputes the backward
        stages instead of saving the ``~n_stages``x dense buffer (memory for
        compute, gradient unchanged). Passing ``event=`` pins the forward
        ``jax_adjoint`` path.
    time_unit : str, optional
        The unit ``t_span`` / ``t_eval`` are in (``"s"``/``"min"``/``"h"``/
        ``"d"``). Default ``None`` uses the plant's native time unit. When
        given, the input times are converted to the native unit for the solve
        and ``solution.t`` is reported back in ``time_unit``.
    event : diffrax.Event, optional
        The low-level single terminating event (used internally by
        :meth:`run_to_steady_state`); pins the forward ``jax_adjoint`` path.
    events : sequence of Event, optional
        Located plant-wide discontinuities (on/off pumps, SBR phase switches,
        dosing on/off, tank-level limits). The monolithic solve is split into
        segments at the firings and ``solution.events_log`` records them.
        Event times (``at_times``, the ``t`` passed to ``cond_fn`` /
        ``apply``, and the log times) use the same unit as ``t_span``.
        Time-only events keep ``jax.grad`` finite and the solve traceable
        under ``jax.jit``; a state event makes the run a forward simulation.
        The compiled solve between events is kept per plant and reused. Mutually exclusive with ``event=`` and with
        ``diff=DifferentiationConfig(method='stable')`` reverse differentiation.
    progress_meter : diffrax.AbstractProgressMeter, optional
        A diffrax progress meter for a long forward solve.
    forward_fast : bool, optional
        Use the lean non-AD forward integrator (no diffrax adjoint machinery):
        a much faster compile + run for a one-off forward solve, but the result
        is NOT differentiable and it needs concrete ``params``/``y0``. It runs
        its own hand-rolled ESDIRK and honours only ``rtol`` / ``atol`` /
        ``integrator.max_steps``; a non-default integrator ``solver`` /
        ``order`` / ``factormax`` / ``dtmax`` / ``colored_jacobian`` is
        **rejected** (rather than silently dropped).

    Returns
    -------
    PlantSolution
    """
    # Resolve the public config objects into the internal primitive locals the
    # rest of this method (and the build helpers) operate on. IntegratorConfig
    # carries the step machinery; DifferentiationConfig carries the AD mode.
    solver = integrator.solver
    factormax = integrator.factormax
    colored_jacobian = integrator.colored_jacobian
    dtmax = integrator.dtmax
    max_steps = int(integrator.max_steps)
    order = int(integrator.order)
    # DifferentiationConfig -> the internal gradient / adjoint routing. A pure
    # decode of (mode, method); the cross-argument combination checks (the
    # single-argument gradient / colored_jacobian validity, the stable-vs-dtmax
    # conflict, and the events / forward_fast combinations) are folded into
    # _validate_solve_args below, so solve reads resolve -> validate ->
    # collect-events -> dispatch.
    gradient, adjoint, adjoint_max_steps, adjoint_low_memory = self._resolve_diff_config(diff)
    self._build_state_layout()
    self._build_parameter_layout()
    if params is None:
        params = self.default_parameters()
    else:
        params = self._coerce_params(params)
    # Initial state: the per-unit defaults, or a caller-supplied warm start
    # (e.g. a previously-computed steady state -- the standard way to start a
    # dynamic-influent run, avoiding the stiff clean-start transient).
    if y0 is None:
        y0 = self.initial_state()
    else:
        y0 = jnp.asarray(y0)
        if y0.shape != (self._total_state_size,):
            raise ValueError(f"y0 has shape {y0.shape}, expected ({self._total_state_size},)")
    # Default atol is a per-component noise floor scaled off the operating
    # magnitudes (the warm start y0 and the per-unit defaults), so a g/m³
    # plant solves without the old fixed 1e-9 forcing the step ceiling.
    if atol is None:
        atol_eff = self._default_atol(y0)
    else:
        atol_eff = _coerce_atol(atol, self._total_state_size)
    # Convert t_span / t_eval from a caller-supplied time_unit into the
    # plant's native (rate-constant) unit; _time_factor scales back on output.
    t_span, t_eval, _time_factor = to_native_time(self.time_unit, time_unit, t_span, t_eval)
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not (t1 > t0):
        raise ValueError(f"t_span end must exceed start; got ({t0}, {t1}).")
    if t_eval is not None:
        t_eval = jnp.asarray(t_eval)

    # Auto-collect located phase-transition events that units declare (an
    # SBRUnit's cycle boundaries), merged with any user-supplied events, so
    # the integrator lands exactly on every phase switch without the caller
    # hand-listing them. cycle_events takes the native-time span. The user's
    # events are in the requested time unit, so convert copies of them to
    # native first (the unit events are already native).
    if events is not None:
        events = events_to_native_time(events, _time_factor)
    _check_inlet_flows(self, t0, t1)
    warn_if_outside_records(self.influents, t0, t1, self.time_unit)
    unit_events: list[Event] = []
    for unit in self.units.values():
        if isinstance(unit, CycleEventSource):
            unit_events.extend(unit.cycle_events(t0, t1))
    if unit_events:
        events = (list(events) + unit_events) if events is not None else unit_events

    # Reject incompatible argument combinations BEFORE any concrete solve work
    # (the affinity/recycle-map probes below call _resolve_flows, which would
    # raise an obscure error on an intentionally-minimal plant).
    self._validate_solve_args(
        events=events,
        event=event,
        integrator=integrator,
        forward_fast=forward_fast,
        gradient=gradient,
        params=params,
        y0=y0,
    )

    # One-time, concrete check that the recycle-flow solve is self-consistent
    # (every flow rule affine in the recycle flows). Skipped under tracing
    # (params/y0 are JAX tracers -- can't compare/warn) and guarded to run
    # once per plant. It only warns, never blocks the solve. Run before the
    # events branch too, so an events-only plant (an SBR/control study) still
    # gets these diagnostics and the cached recycle map.
    #
    # LIMITATION: this probes affinity only at (t0, y0). A unit with a
    # piecewise-linear flow rule -- a ThresholdSplitter (influent
    # bypass) or a StorageTank level-gated bypass -- that is on one side of
    # its kink at t0 but crosses it *later* in a dynamic run is NOT caught:
    # _resolve_flows then linearises across the kink at that time with no
    # warning. The resolved flows stay exact only while every unit remains in
    # the affine regime it occupied at t0. (Re-probing at sampled times would
    # close this; deferred as a known limitation -- see issue #255.)
    self._run_one_time_guards(t0, params, y0)

    if events is not None:
        # (argument combinations validated above, before the concrete checks)
        return self._solve_with_events(
            t0,
            t1,
            t_eval,
            params,
            y0,
            events,
            rtol=rtol,
            atol=atol_eff,
            dtmax=dtmax,
            adjoint=adjoint,
            max_steps=max_steps,
            time_factor=_time_factor,
            time_unit=time_unit,
            order=order,
            factormax=factormax,
            solver=solver,
        )

    if gradient == "auto":
        # A concrete forward solve takes the fast cached jax_adjoint path; a
        # solve under reverse-mode differentiation (params/y0 are tracers)
        # takes the cap-free stable_adjoint, so a stiff plant gradient is
        # finite by default. event=/adjoint=/dtmax= are jax_adjoint-only, so
        # their presence pins jax_adjoint.
        differentiating = any(isinstance(v, jax.core.Tracer) for v in (params, y0) if v is not None)
        pin_jax = event is not None or adjoint is not None or dtmax is not None
        gradient = "jax_adjoint" if (pin_jax or not differentiating) else "stable_adjoint"

    if gradient == "stable_adjoint":
        return self._solve_stable_adjoint(
            t0,
            t1,
            t_eval,
            params,
            y0,
            rtol=rtol,
            atol=atol_eff,
            solver=solver,
            factormax=factormax,
            colored_jacobian=colored_jacobian,
            order=order,
            adjoint=adjoint,
            dtmax=dtmax,
            event=event,
            adjoint_max_steps=adjoint_max_steps,
            adjoint_low_memory=adjoint_low_memory,
            time_factor=_time_factor,
            time_unit=time_unit,
        )

    if forward_fast:
        sol = self._solve_forward_fast(
            t0,
            t1,
            t_eval,
            params,
            y0,
            rtol=rtol,
            atol=atol_eff,
            max_steps=max_steps,
            time_factor=_time_factor,
            time_unit=time_unit,
        )
        if sol is not None:
            return sol

    return self._solve_jax_adjoint(
        t0,
        t1,
        t_eval,
        params,
        y0,
        rtol=rtol,
        atol=atol_eff,
        event=event,
        adjoint=adjoint,
        dtmax=dtmax,
        max_steps=max_steps,
        progress_meter=progress_meter,
        solver=solver,
        factormax=factormax,
        order=order,
        colored_jacobian=colored_jacobian,
        time_factor=_time_factor,
        time_unit=time_unit,
    )


def _resolve_diff_config(diff):
    """Decode a public :class:`DifferentiationConfig` into internal routing.

    Pure translation of ``(diff.mode, diff.method)`` into the internal
    ``(gradient, adjoint, adjoint_max_steps, adjoint_low_memory)`` primitives
    the solve dispatch operates on:

    * ``reverse`` + ``stable``        -> ``gradient="auto"`` (a concrete forward
      solve routes to the cached fast path; a differentiated one to the
      cap-free discrete adjoint)
    * ``reverse`` + ``through_solve`` -> ``gradient="jax_adjoint"`` (differentiate
      through the diffrax solve)
    * ``forward`` + ``through_solve`` -> ``gradient="jax_adjoint"`` with a
      forward-mode ``adjoint`` (``jvp`` / ``jacfwd`` flow through the solve)

    Raises ``ValueError`` for a malformed config (bad ``mode`` / ``method``) or
    the ``forward`` + ``stable`` augmented variational solve, which
    :meth:`solve` does not serve (call :meth:`solve_sensitivity` /
    :meth:`dynamic_sensitivity`). Cross-argument combination checks live in
    :meth:`_validate_solve_args`.
    """
    diff.validated()
    if diff.mode == "forward" and diff.method == "stable":
        raise ValueError(
            "diff=DifferentiationConfig(mode='forward', method='stable') is "
            "the augmented variational solve; call plant.solve_sensitivity "
            "(or plant.dynamic_sensitivity) for it, not plant.solve."
        )
    # ``method="stable"`` routes through the plant's own ``"auto"`` dispatch
    # (concrete forward solve -> cached fast path; a differentiated one ->
    # cap-free discrete adjoint) rather than the canonical "stable_adjoint";
    # otherwise differentiate through the diffrax solve. The forward-mode
    # adjoint object (for forward + through_solve) is the config's.
    gradient = "auto" if diff.method == "stable" else "jax_adjoint"
    adjoint = diff.reactor_adjoint()
    return gradient, adjoint, int(diff.adjoint_max_steps), bool(diff.adjoint_low_memory)


def _validate_solve_args(self, *, events, event, integrator, forward_fast, gradient, params, y0):
    """Reject incompatible :meth:`solve` argument combinations up front.

    Raises ``ValueError`` for: an out-of-range ``gradient`` /
    ``integrator.colored_jacobian``; the reverse-``stable`` gradient (routed as
    ``gradient='auto'``) paired with an ``integrator.dtmax`` cap, which it
    cannot honour (it forms its own cap-free discrete adjoint -- ``dtmax`` is
    meaningful only when differentiating *through* the diffrax solve,
    ``method='through_solve'``); ``events=`` with the low-level ``event=`` or
    with ``integrator.solver`` / ``colored_jacobian=True`` (the segmented
    located-event solve manages its own integrator); and ``forward_fast`` with
    ``events=`` / the internal ``"stable_adjoint"`` route / traced inputs (it is a
    non-differentiable concrete-only path) / a non-default integrator
    ``solver`` / ``order`` / ``factormax`` / ``dtmax`` / ``colored_jacobian``
    (it honours only ``rtol`` / ``atol`` / ``max_steps``).
    """
    # Single-argument validity + the stable-vs-dtmax cross-check, folded here
    # from solve() so every combination check lives in one place.
    if gradient not in ("auto", "jax_adjoint", "stable_adjoint"):
        raise ValueError(
            f"gradient must be 'auto', 'jax_adjoint' or 'stable_adjoint'; got {gradient!r}."
        )
    if integrator.colored_jacobian not in (True, False, "auto"):
        raise ValueError(
            f"colored_jacobian must be True, False or 'auto'; got {integrator.colored_jacobian!r}."
        )
    # gradient == "auto" is exactly reverse+stable (see _resolve_diff_config):
    # it forms its own cap-free discrete adjoint and controls its own steps, so
    # an explicit dtmax cap alongside it is a usage error. Catching it here
    # keeps the later "auto" routing from silently treating dtmax as a request
    # to pin the jax_adjoint path.
    if gradient == "auto" and integrator.dtmax is not None:
        raise ValueError(
            "diff=DifferentiationConfig(method='stable') forms its own discrete "
            "adjoint and controls its own steps; do not also pass "
            "integrator=IntegratorConfig(dtmax=...) (the stable method is "
            "cap-free). Use method='through_solve' if you need the dtmax cap."
        )

    if events is not None:
        if event is not None:
            raise ValueError(
                "pass either events= (the user-facing located-event API) or "
                "the low-level event= (a single diffrax terminating event), "
                "not both."
            )
        # The located-event solve runs a segmented solve through the diffrax
        # path (with the resolved adjoint), so the cap-free reverse *stable*
        # discrete adjoint -- which has no segmented form -- cannot back it.
        # ``method="stable"`` is the default and routes a concrete events solve
        # to the segmented jax_adjoint path fine; only an explicit request to
        # differentiate the events solve via the stable discrete adjoint is the
        # error, which the runtime trace below reports. Here we reject the two
        # integrator opt-ins the segmented solve cannot honour.
        if integrator.solver is not None or integrator.colored_jacobian is True:
            raise ValueError(
                "integrator.solver / colored_jacobian=True are not supported "
                "with events=; the located-event solve manages its own "
                "integrator. Drop them."
            )

    # forward_fast is the lean non-AD forward integrator: no diffrax adjoint /
    # optimistix / lineax machinery, so the result is NOT differentiable and it
    # needs concrete inputs to build its colored-Jacobian pattern once.
    if forward_fast:
        if events is not None:
            raise ValueError(
                "forward_fast is not supported with events=; the located-event "
                "solve manages its own integrator. Drop one."
            )
        if gradient == "stable_adjoint":
            raise ValueError(
                "forward_fast is a non-differentiable forward path; it is "
                f"incompatible with the stable discrete adjoint ({STABLE_REVERSE_DIFF}). "
                "For a reverse-mode gradient use the default solve."
            )
        if any(isinstance(v, jax.core.Tracer) for v in (params, y0)):
            raise ValueError(
                "forward_fast requires concrete params/y0 -- it is a non-AD "
                "fast path that is not differentiable and cannot be traced "
                "(no jax.grad / jax.jit). For gradients or jit use the default "
                "solve (which routes through the differentiable diffrax path)."
            )
        # forward_fast runs its own hand-rolled ESDIRK (Kvaerno3 with an
        # internal colored Jacobian and the diffrax default step growth) and
        # threads only rtol / atol / max_steps. The integrator's solver / order
        # / factormax / dtmax / colored_jacobian knobs cannot reach it, so a
        # *non-default* value would be silently dropped -- reject it instead,
        # mirroring the events= path. (max_steps IS honoured, so it stays
        # tunable; a value left at its default signals no intent to deviate.)
        default = IntegratorConfig()
        dropped = [
            name
            for name in ("solver", "order", "factormax", "dtmax", "colored_jacobian")
            if getattr(integrator, name) != getattr(default, name)
        ]
        if dropped:
            raise ValueError(
                "forward_fast cannot honour integrator "
                f"{' / '.join(dropped)}: it runs its own hand-rolled ESDIRK "
                "(Kvaerno3 with an internal colored Jacobian and the diffrax "
                "default step growth) and threads only rtol / atol / max_steps. "
                "Drop the non-default integrator setting(s), or use the default "
                "solve (which honours them)."
            )


def _run_one_time_guards(self, t0, params, y0):
    """One-time, concrete recycle diagnostics + map-constancy caching.

    Skipped under tracing (``params``/``y0`` are JAX tracers). Runs the
    affinity / sweep-convergence warnings once per plant, and probes whether
    the recycle concentration map ``M`` and flow map ``A`` are
    state-independent (so they can be cached once per solve). Only warns;
    never blocks the solve.
    """
    if not any(isinstance(v, jax.core.Tracer) for v in (params, y0)):
        if not self._flow_affinity_checked:
            self._flow_affinity_checked = True
            self._recycle._resolve_flows(
                jnp.asarray(t0),
                params,
                states=self._split_state(y0),
                check_affine=True,
            )
        # Companion diagnostic: does the fixed-pass recycle concentration
        # sweep converge at recycle_passes? (warns once, never blocks.)
        if not self._recycle_convergence_checked:
            self._recycle_convergence_checked = True
            self._recycle._check_recycle_convergence(jnp.asarray(t0), self._split_state(y0), params)
        # Is the recycle affine map M state-independent? If so, it can be
        # precomputed once per solve and reused -- a large per-RHS saving.
        if self._recycle._recycle_map_constant is None:
            self._recycle._check_recycle_map_constant(jnp.asarray(t0), y0, params)
        # Is the recycle *flow* map A state-independent? If so it is cached
        # once per solve too, skipping the per-RHS flow probe.
        if self._recycle._flow_map_constant is None:
            self._recycle._check_flow_map_constant(jnp.asarray(t0), y0, params)
        # (The colored backward Jacobian builder is derived in the
        # stable_adjoint branch below, only when that path is actually taken,
        # so a forward jax_adjoint solve never builds it.)


def _finalize_solution(self, ts, ys, *, time_factor, time_unit, events_log=None):
    """Wrap a solved ``(ts, ys)`` pair into a :class:`PlantSolution`.

    Rescales the native solver times back to the caller's requested unit
    (``time_factor``, native -> requested) and tags the requested unit so the
    solution reports its times in it. Shared by every solve-dispatch path so the
    rescale + construction lives in one place rather than being repeated apiece.
    """
    if time_factor != 1.0:
        ts = ts / time_factor  # native -> requested unit
    return PlantSolution(
        t=ts,
        state=ys,
        plant=self,
        events_log=events_log,
        _requested_time_unit=time_unit,
    )


def _solve_stable_adjoint(
    self,
    t0,
    t1,
    t_eval,
    params,
    y0,
    *,
    rtol,
    atol,
    solver,
    factormax,
    colored_jacobian,
    order,
    adjoint,
    dtmax,
    event,
    adjoint_max_steps,
    adjoint_low_memory,
    time_factor,
    time_unit,
):
    """The cap-free reverse-mode discrete-adjoint solve (the internal ``"stable_adjoint"`` route).

    A robust adaptive ESDIRK forward paired with the per-step transposed-solve
    discrete adjoint over the saved trajectory, which stays finite at any step
    size. ``solve`` routes here only under reverse-mode differentiation, so
    it runs uncached: the ``custom_vjp`` composes with the enclosing pass.
    """
    if adjoint is not None or dtmax is not None:
        raise ValueError(
            f"The stable discrete adjoint ({STABLE_REVERSE_DIFF}) forms its own "
            "discrete adjoint and controls its own steps; do not also pass "
            "adjoint= or dtmax=."
        )
    if event is not None:
        raise ValueError(
            "event= (e.g. a steady-state terminating event) is only "
            "supported on the forward path that differentiates through the "
            f"solve ({FORWARD_THROUGH_SOLVE_DIFF} or method='through_solve')."
        )
    # solver=/factormax= ARE supported here: the discrete adjoint builds
    # its backward from the forward solver's Butcher tableau generically
    # (any ESDIRK with a tableau -- e.g. the cheaper 4-stage Kvaerno3 in
    # place of the default 7-stage Kvaerno5), and factormax caps the
    # per-step growth of its PID controller, exactly as on the forward
    # path. A non-ESDIRK solver raises a clear error from the tableau
    # extraction. ``solver=None`` keeps the default Kvaerno5.
    # colored_jacobian colors the per-step df/dy Jacobian build in the
    # BACKWARD pass (its dominant cost), exact (machine-precision) vs dense.
    # It is enabled deterministically by plant SIZE (the reliable signal --
    # see _COLORED_BACKWARD_MIN_STATES), not a build-time benchmark, so no
    # concrete probe solve is needed: the builder is derived from the plant's
    # default operating point and so builds even here under the gradient's
    # own trace. ``True`` forces it on (any size), ``"auto"`` enables it past
    # the size gate (_auto_colored_backward, the same gate
    # colored_jacobian_decision reports), ``False`` never colors.
    n_states = int(y0.shape[0])
    want_colored = colored_jacobian is True or (
        colored_jacobian == "auto"
        and _auto_colored_backward(n_states, self._COLORED_BACKWARD_MIN_STATES) == "colored"
    )
    if want_colored and self._colored._adjoint_builder is None:
        self._colored.adjoint_jacobian_builder(t0, rtol, atol, y0=y0)
    cab = self._colored._adjoint_builder
    use_colored = bool(want_colored and cab is not None and cab[2])  # built + guard ok
    jac_builder = cab[0] if use_colored else None
    # The infrastructure to also color the *forward* solve's per-step
    # implicit Jacobian is in place (``cab[4]`` is the ColoredVeryChord for
    # the autonomized forward RHS, and ``esdirk_adjoint_solve`` accepts it
    # as ``forward_root_finder``), but it is NOT auto-enabled: unlike the
    # colored *backward* (which feeds J directly into the transposed solve,
    # exact on a superset pattern), the colored forward feeds J into an
    # iterative chord whose decoupled-Newton convergence point depends on
    # the J approximation, so a colored-vs-dense difference shifts the
    # forward trajectory at the ~Newton-tolerance level (~1e-4) -- it would
    # break the bit-identical ``colored_jacobian=True`` == dense invariant.
    # Enabling it as a default needs the structural pattern reconciled so
    # the colored and dense forward chords converge identically.
    fwd_root_finder = None
    # Cap-free reverse-mode gradient through the stiff plant solve: the
    # forward is a robust adaptive ESDIRK solve and the reverse is the
    # per-step transposed-solve discrete adjoint over the saved trajectory,
    # which stays finite at any step size. ``solve`` routes here only under
    # reverse-mode differentiation (params or y0 traced), so the adjoint's
    # ``custom_vjp`` is traced directly into the outer computation -- routing
    # it through an inner ``jax.jit`` would not compose with the outer
    # reverse-mode pass -- and an enclosing ``jax.jit`` amortizes the compile.
    # The recycle probe is still hoisted via the cached-map primal RHS, and
    # the colored backward Jacobian builder is used when requested. The
    # discrete-adjoint backward scan walks a saved trajectory whose length is
    # adjoint_max_steps (DifferentiationConfig.adjoint_max_steps); the
    # forward solve must also fit in that buffer, so this single budget serves
    # both roles here.
    adj_steps = adjoint_max_steps
    with friendly_solve_errors(adj_steps, what="plant solve"):
        ys = self._esdirk_stable_adjoint(
            y0,
            params,
            t0,
            t1,
            t_eval,
            rtol=rtol,
            atol=atol,
            max_steps=adj_steps,
            jacobian_builder=jac_builder,
            forward_root_finder=fwd_root_finder,
            solver=solver,
            factormax=factormax,
            order=order,
            low_memory=adjoint_low_memory,
        )
    if t_eval is None:
        ts = jnp.asarray([t1])
        ys = ys[None, :]
    else:
        ts = jnp.asarray(t_eval)
    return self._finalize_solution(ts, ys, time_factor=time_factor, time_unit=time_unit)


def _solve_forward_fast(
    self,
    t0,
    t1,
    t_eval,
    params,
    y0,
    *,
    rtol,
    atol,
    max_steps,
    time_factor,
    time_unit,
):
    """The lean non-AD forward integrator (``forward_fast=True``).

    Returns the :class:`PlantSolution` on success, or ``None`` when the
    colored-Jacobian start-state guard fails (after warning) so the caller
    falls back to the differentiable diffrax forward path.
    """
    # Forward-fast: the lean non-AD integrator (no diffrax adjoint / optimistix
    # / lineax). It needs the colored-Jacobian pattern (for a cheap per-step
    # Jacobian); build + guard it, and fall back to the diffrax path if the
    # guard fails. Concrete-only (validated above). Its compiled solve is
    # cached per instance like the diffrax path.
    rf = self._colored.forward_root_finder(t0, y0, params, rtol, atol)
    if rf is not None:
        te = t_eval if t_eval is not None else jnp.asarray([float(t1)])
        fkey = (
            "forward_fast",
            t0,
            t1,
            tuple(te.shape),
            concrete_settings_key(rtol, atol, None, None, max_steps),
            self._recycle._recycle_map_constant,
        )
        jitted = self._solve_cache.get("jit", fkey)
        if jitted is None:
            jitted = self._build_jitted_forward_fast(
                rf, t0, t1, rtol=rtol, atol=atol, max_steps=max_steps
            )
            self._solve_cache.put("jit", fkey, jitted)
        with friendly_solve_errors(max_steps, what="plant forward_fast solve"):
            ts, ys, result = jitted(y0, params, te)
            ts, ys = check_solve_result(
                result, (ts, ys), max_steps=max_steps, what="plant forward_fast solve"
            )
        return self._finalize_solution(ts, ys, time_factor=time_factor, time_unit=time_unit)
    warnings.warn(
        "forward_fast: the colored-Jacobian start-state guard failed; "
        "falling back to the diffrax forward path for this plant.",
        RuntimeWarning,
        stacklevel=2,
    )


def _solve_jax_adjoint(
    self,
    t0,
    t1,
    t_eval,
    params,
    y0,
    *,
    rtol,
    atol,
    event,
    adjoint,
    dtmax,
    max_steps,
    progress_meter,
    solver,
    factormax,
    order,
    colored_jacobian,
    time_factor,
    time_unit,
):
    """The diffrax solve (the internal ``"jax_adjoint"`` route) -- the default path.

    Runs the plant under the standard diffrax integrator: a plain concrete
    forward integration, or -- when wrapped in ``jax.grad`` / ``jacfwd`` --
    differentiation *through* the solve via ``adjoint`` (reverse or forward
    mode). Distinct from the augmented ``[y; S]`` variational solve, which is
    a separate method (:meth:`solve_sensitivity`). Supports an optional colored
    per-step implicit Jacobian. The compiled solve is cached per instance and
    reused across repeat solves; an event-terminated or traced call bypasses
    the cache.
    """
    # The compiled solve is cached per instance
    # and reused across repeat solves of this plant (see ``_jit_cache``). An
    # event-terminated solve (e.g. run_to_steady_state) is not cached -- the
    # event closure varies and that path is run once -- and a traced call
    # (settings key None) bypasses the cache. Caching does not change the
    # first solve; it only avoids recompiling on subsequent ones.
    # Colored-AD Jacobian: reconfigure the solver to materialise the per-step
    # implicit Jacobian by sparse column compression (a large saving on the
    # dense flowsheet Jacobian). Built+guarded once per plant; falls back to
    # the dense solver if the start-state guard fails or it can't be built
    # (a first traced solve). Numerically identical when the pattern is a
    # superset, so it composes with solver=/factormax= and the cached map.
    # Forward coloring is enabled by an explicit ``colored_jacobian=True``.
    # The default ``"auto"`` governs only the (exact) stable_adjoint BACKWARD
    # decision; it leaves the forward solve dense, because forward coloring
    # swaps the implicit linear solver and so is not guaranteed bit-identical
    # -- making it the all-solves default needs its own full-suite validation
    # (a deliberate follow-up). ``True`` opts into both.
    if colored_jacobian is True:
        solver = self._colored.jacobian_solver(solver, t0, y0, params, rtol, atol)
    colored_active = (
        colored_jacobian is True
        and self._colored.forward_root_finder(t0, y0, params, rtol, atol) is not None
    )

    settings = concrete_settings_key(rtol, atol, adjoint, dtmax, max_steps)
    sig = (t0, t1, None if t_eval is None else tuple(t_eval.shape))
    # The solver is keyed by class name: a fresh stock solver instance (the
    # common case) shares the cache with another of the same class, while a
    # different class (Kvaerno3 vs the default Kvaerno5) keys separately. A
    # custom-*configured* instance of an otherwise-default class would share
    # the default's entry -- documented on the ``solver=`` argument.
    solver_key = None if solver is None else type(solver).__name__
    # The recycle-map reuse flags are part of the compiled solve (they select
    # the cached-M / cached-MT code path). They are set once per instance
    # before the first build, so they are constant for a given plant -- keying
    # on them only guards the rare case where the very first solve was traced
    # (flags still None) and a later concrete solve sets them.
    recycle_key = self._recycle.cache_flags()
    # A progress meter is a one-off diagnostic (and carries host state), so it
    # bypasses the compiled-solve cache rather than being keyed into it.
    cache_key = (
        None
        if (settings is None or event is not None or progress_meter is not None)
        else (sig, settings, solver_key, factormax, recycle_key, colored_active, order)
    )
    jitted = self._solve_cache.get("jit", cache_key)
    if jitted is None:
        jitted = self._build_jitted_solve(
            t0,
            t1,
            t_eval is not None,
            event=event,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
            dtmax=dtmax,
            max_steps=max_steps,
            progress_meter=progress_meter,
            solver=solver,
            factormax=factormax,
            order=order,
        )
        if cache_key is not None:
            # An adjoint keyed by identity (see _adjoint_key) is kept alive with it.
            self._solve_cache.put("jit", cache_key, jitted, adjoint)

    with friendly_solve_errors(max_steps, what="plant solve"):
        if t_eval is None:
            ts, ys, result = jitted(y0, params)
        else:
            ts, ys, result = jitted(y0, params, t_eval)
        ts, ys = check_solve_result(result, (ts, ys), max_steps=max_steps, what="plant solve")
    return self._finalize_solution(ts, ys, time_factor=time_factor, time_unit=time_unit)


def _solve_with_events(
    self,
    t0,
    t1,
    t_eval,
    params,
    y0,
    events,
    *,
    rtol,
    atol,
    dtmax,
    adjoint,
    max_steps,
    time_factor,
    time_unit,
    order=5,
    factormax=None,
    solver=None,
):
    """Run the monolithic plant solve with located events (the ``events=``
    path).

    Hands the assembled plant RHS to the segmented driver
    (:func:`~aquakin.integrate.events.run_segmented`), which splits the solve
    at the event times and applies their resets to the flat plant state between
    segments. Every segment runs one compiled solve whose bounds, save grid,
    initial state, parameters and recycle maps are runtime arguments; it is kept
    in the per-instance ``_jit_cache`` (:func:`_event_segment_solver`), so a
    repeat solve of the same plant and settings compiles nothing. With only time
    events nothing depends on a concrete value, so the solve differentiates and
    traces under ``jax.jit``.

    The state-invariant recycle maps are built once per solve (from ``params``,
    so a time-event gradient still flows through them) at ``(t0, y0)`` and
    reused by every segment -- the same per-RHS saving the forward solve gets;
    ``None`` (the probe path) when a map is state-coupled or the constancy check
    has not run.
    """
    maps = self._recycle.cached_maps(jnp.asarray(t0), self._split_state(y0), params)
    settings = SegmentSettings(
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        dtmax=dtmax,
        adjoint=adjoint,
        order=order,
        factormax=factormax,
        solver=solver,
    )
    solve_segment = self._event_segment_solver(settings, events, has_t_eval=t_eval is not None)

    with friendly_solve_errors(max_steps, what="plant solve"):
        res = run_segmented(
            solve_segment,
            maps,
            y0,
            params,
            t_span=(t0, t1),
            t_eval=t_eval,
            events=events,
            settings=settings,
        )
    log = res.log
    if time_factor != 1.0:  # native -> requested unit
        log = [(t / time_factor, name) for t, name in log]
    return self._finalize_solution(
        res.ts, res.ys, time_factor=time_factor, time_unit=time_unit, events_log=log
    )


def _event_segment_solver(self, settings, events, *, has_t_eval):
    """The compiled one-segment solve of the ``events=`` path, cached per plant.

    Keyed, in ``_jit_cache``, by everything the compiled program depends on:
    the solver settings (``rtol``, the ``atol`` values, the adjoint, ``dtmax``,
    ``max_steps``, ``order``, ``factormax``, the solver class), the root-find
    tolerances, whether a ``t_eval`` grid is reported, the state events located
    in-solve (:func:`~aquakin.integrate.events.segment_signature`) and the
    recycle map-constancy verdicts the RHS is specialised on. The segment
    bounds, the grid, ``y0``, ``params`` and the maps are runtime arguments, so
    they are not keyed (JAX's own cache covers their shapes); the plant's units,
    wiring and conditions are fixed for the life of the cache, which
    :meth:`_structure_changed` and the condition mutators drop. Under tracing
    (a traced ``atol``) the key cannot be formed and the solver is built
    uncached.

    Parameters
    ----------
    settings : SegmentSettings
        The integrator settings of the solve.
    events : sequence of Event
        The (native-time) events of the solve.
    has_t_eval : bool
        Whether the solve reports a ``t_eval`` grid.

    Returns
    -------
    callable
        The jitted ``solve_segment`` from
        :func:`~aquakin.integrate.events.build_segment_solver`.
    """
    key = settings.cache_key()
    if key is not None:
        key = (
            "events",
            key,
            has_t_eval,
            segment_signature(events),
            self._recycle.cache_flags(),
        )
        cached = self._solve_cache.get("jit", key)
        if cached is not None:
            return cached[0]
    rhs_with_maps = self._recycle.rhs_with_maps
    solve_segment = build_segment_solver(
        rhs_with_maps, settings, root_events_of(events), has_t_eval=has_t_eval
    )
    if key is not None:
        # An adjoint keyed by identity (see _adjoint_key) is kept alive with the
        # entry so its id cannot be reused by another object while it exists.
        self._solve_cache.put("jit", key, (solve_segment, settings.adjoint))
    return solve_segment


def _build_jitted_solve(
    self,
    t0,
    t1,
    has_t_eval,
    *,
    event,
    rtol,
    atol,
    adjoint,
    dtmax,
    max_steps,
    progress_meter=None,
    solver=None,
    factormax=None,
    order=5,
):
    """Build the jit-compiled forward solve for one call signature.

    The returned ``_solve`` closes over the (static) plant RHS and the solver
    settings; ``y0``/``params``/``t_eval`` are its runtime arguments, so the
    same compiled solve serves any parameter or initial-state vector of the
    cached shapes (the parameter-sweep case). ``t_eval`` is a runtime
    argument (not baked in) so different save times of the same shape reuse
    one compile.

    When the recycle affine map ``M`` is state-independent
    (:meth:`_check_recycle_map_constant`), it is computed **once** per solve
    from the runtime ``params`` (so the gradient still flows and a parameter
    sweep stays correct) and passed into every RHS call as ``recycle_map=``,
    collapsing the per-RHS recycle resolution from ``n_recycle_edges + 1``
    sweeps to one. Bit-identical to probing it every call.
    """
    t0_arr = jnp.asarray(float(t0))

    def make_rhs(y0, params):
        rhs, _ = self._recycle.cached_rhs(t0_arr, self._split_state(y0), params)
        return rhs

    kw = dict(
        t0=t0,
        t1=t1,
        rtol=rtol,
        atol=atol,
        adjoint=adjoint,
        dtmax=dtmax,
        max_steps=max_steps,
        event=event,
        progress_meter=progress_meter,
        solver=solver,
        factormax=factormax,
        order=order,
    )

    if has_t_eval:

        @jax.jit
        def _solve(y0, params, t_eval):
            sol = _run_diffeqsolve(
                make_rhs(y0, params),
                y0=y0,
                args=params,
                saveat=diffrax.SaveAt(ts=t_eval),
                **kw,
            )
            return sol.ts, sol.ys, sol.result

        return _solve

    @jax.jit
    def _solve(y0, params):
        sol = _run_diffeqsolve(
            make_rhs(y0, params),
            y0=y0,
            args=params,
            saveat=diffrax.SaveAt(t1=True),
            **kw,
        )
        return sol.ts, sol.ys, sol.result

    return _solve


def _build_jitted_forward_fast(self, root_finder, t0, t1, *, rtol, atol, max_steps):
    """Build the lean non-AD forward solve for one signature.

    Wraps :func:`~aquakin.integrate.forward_solve.forward_solve` -- a plain
    ``lax.while_loop`` adaptive ESDIRK with an Anderson-accelerated
    simplified Newton + colored Jacobian + direct ``lu_factor``/``lu_solve``,
    no diffrax adjoint /
    optimistix / lineax. The per-step Jacobian is still colored forward-mode
    AD (the same exact matrix the differentiable path uses), but the solve is
    NOT differentiable. ``y0``/``params``/``t_eval`` are runtime arguments so
    one compile serves a parameter sweep; the colored pattern of
    ``root_finder`` (built+guarded before this is called) is baked into the
    closure. The recycle map is precomputed once per solve from ``params``
    (the cached-M path).
    """
    from aquakin.integrate.colored_jacobian import state_jacobian
    from aquakin.integrate.forward_solve import StepControl, forward_solve

    t0a = jnp.asarray(float(t0))

    @jax.jit
    def _solve(y0, params, t_eval):
        rhs, _ = self._recycle.cached_rhs(t0a, self._split_state(y0), params)

        ys, result = forward_solve(
            rhs,
            state_jacobian(rhs, root_finder),
            y0,
            params,
            float(t0),
            float(t1),
            t_eval,
            rtol=rtol,
            atol=atol,
            control=StepControl(max_steps=max_steps),
        )
        return t_eval, ys, result

    return _solve


def _esdirk_stable_adjoint(
    self,
    y0,
    params,
    t0,
    t1,
    t_eval,
    *,
    rtol,
    atol,
    max_steps,
    jacobian_builder=None,
    forward_root_finder=None,
    solver=None,
    factormax=None,
    order=5,
    low_memory=False,
):
    """Cap-free reverse-mode plant solve with the cached recycle map hoisted
    out of the backward pass.

    Caches the state-invariant recycle map once per solve and reuses it on the
    *primal* RHS (the forward solve + every backward stage / df/dy Jacobian),
    so the expensive per-call recycle probe is lifted out of the hot
    reverse-adjoint loop. The plain ``rhs`` -- which recomputes the map from
    ``params`` -- still drives the df/dtheta vjp, so the dM/dtheta term
    (nonzero only for flow-setpoint params) is exact; the gradient is
    bit-identical to probing on every call, just faster.

    The cached map must be ``stop_gradient``'d: it is a params-derived value
    closed over inside the discrete-adjoint custom VJP, and its parameter
    dependence is accounted for by the vjp's ``rhs``, not by this closure.
    When the map is not state-invariant (``_recycle_map_constant`` not True,
    e.g. a scheduled pump) the cache is ``None`` and the primal falls back to
    ``rhs`` (probe per call -- correct, just unoptimised). Shared by the
    cached jit closure and the under-trace gradient path so both optimise.
    """

    def rhs(t, y, args):
        return self._rhs(t, y, args)

    # Both cached maps (M and the flow map A) are params-derived values;
    # stop_gradient them before closing over them -- their parameter
    # dependence is the param-vjp's job, which keeps the map-recomputing
    # `rhs` (both maps left None) so a flow-setpoint gradient still captures
    # dM/dtheta and dA/dtheta. The cached primal is used only when M is
    # cached: a constant A alone does not switch it on.
    primal_rhs, (rmap, _) = self._recycle.cached_rhs(
        jnp.asarray(t0), self._split_state(y0), params, stop_gradient=True
    )
    if rmap is None:
        primal_rhs = None

    return esdirk_adjoint_solve(
        rhs,
        y0,
        params,
        (t0, t1),
        t_eval,
        solver=solver,
        order=order,
        factormax=factormax,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        time_dependent=True,
        primal_rhs=primal_rhs,
        jacobian_builder=jacobian_builder,
        forward_root_finder=forward_root_finder,
        low_memory=low_memory,
    )
