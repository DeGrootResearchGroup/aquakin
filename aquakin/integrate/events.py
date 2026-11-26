"""General event / discontinuity handling for the reactor and plant solves.

A plain ODE solve is continuous; many real operations are not. On/off pumps,
SBR fill/react/settle/decant phase switches, relay and saturating controllers,
dosing on/off, and tank-level limits all introduce a **discontinuity** -- either
at a *known time* (a phase schedule) or when a *state crosses a threshold* (a
level limit, a relay setpoint). Encoding these by smoothing or ``searchsorted``
gathers mislocates the switch; this module locates it exactly and applies a
**state reset / mode switch** there, then continues the solve.

Two event kinds are supported, distinguished by how the switch time is found:

* **Time events** (``Event(at_times=[...])``) fire at known times. The segment
  boundaries are static, so the solve is a fixed sequence of differentiable
  sub-solves -- ``jax.grad`` flows through the whole evented solve (the SBR /
  scheduled-dosing case). This is the AD-safe path.
* **State (root-crossing) events** (``Event(cond_fn=...)``) fire when a scalar
  ``cond_fn(t, y, args)`` crosses zero, located by a root find. The number of
  firings is data-dependent, so this path is an **eager forward driver** (it
  re-solves segment by segment) and is not differentiable through the switch;
  use a smoothed condition where a gradient through the threshold is required.

Each event optionally carries an ``apply(t, y, args) -> y`` reset that produces
the post-event state (a pump turning a flow on, an SBR decant removing volume, a
controller latching), and a ``terminal`` flag to stop the solve when it fires.
The driver (:func:`solve_with_events`) is generic over the right-hand side, so
both :meth:`BatchReactor.solve` and :meth:`Plant.solve` expose it via an
``events=`` argument.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from aquakin.integrate._common import (
    _run_diffeqsolve,
    concrete_settings_key,
    friendly_solve_errors,
)
from aquakin.integrate._solver import check_solve_result

# Relative tolerance (scaled by the span) for deciding a save time coincides with
# a segment boundary, so it is assigned to the segment ending there (its
# pre-reset value) rather than duplicated across the reset.
_BOUNDARY_ASSIGN_RTOL = 1e-9


@dataclass
class Event:
    """A located discontinuity in a solve: a time or state trigger + a reset.

    Exactly one trigger must be given:

    * ``at_times`` -- a sequence of times (in the solve's time unit) at which the
      event fires. AD-safe: the segment boundaries are static. A time equal to
      the start of the solve fires on the initial state, before integration.
    * ``cond_fn`` -- a scalar function ``cond_fn(t, y, args)`` whose zero crossing
      (in ``direction``) is the event. Located by a root find; forward-simulation
      only (the firing count is data-dependent).

    Parameters
    ----------
    cond_fn : callable, optional
        Root-crossing condition ``cond_fn(t, y, args) -> scalar``. ``y`` is the
        full state vector (the reactor concentration vector, or the flat plant
        state); ``args`` is the solve's parameter argument. ``t`` is in the same
        time unit as ``at_times``: the solve's time unit (for
        :meth:`BatchReactor.solve` with ``time_unit=``, that unit; otherwise the
        model's native unit). The same holds for the ``t`` passed to ``apply``.
    at_times : sequence of float, optional
        Times at which a time event fires.
    direction : int, optional
        For a ``cond_fn`` event: ``+1`` fires only on an upward crossing
        (negative→positive), ``-1`` only downward, ``0`` (default) either way.
    apply : callable, optional
        State reset ``apply(t, y, args) -> y_new`` producing the post-event state
        (same shape as ``y``). ``None`` (default) leaves the state unchanged --
        useful for a pure ``terminal`` detector.
    terminal : bool, optional
        If True, the solve stops when this event fires (after applying ``apply``).
    name : str, optional
        Label for the event log (defaults to ``event{i}``).
    """

    cond_fn: Callable | None = None
    at_times: Sequence[float] | None = None
    direction: int = 0
    apply: Callable | None = None
    terminal: bool = False
    name: str | None = None

    def __post_init__(self):
        has_cond = self.cond_fn is not None
        has_times = self.at_times is not None
        if has_cond == has_times:
            raise ValueError(
                "an Event needs exactly one trigger: cond_fn (state event) OR "
                "at_times (time event)."
            )
        if self.direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 or +1.")
        if has_times:
            ts = [float(t) for t in self.at_times]
            if any(b <= a for a, b in zip(ts, ts[1:])):
                raise ValueError("at_times must be strictly increasing.")
            self.at_times = ts

    @property
    def is_time_event(self) -> bool:
        return self.at_times is not None


@dataclass
class EventedResult:
    """Output of :func:`solve_with_events`.

    Attributes
    ----------
    ts : jnp.ndarray
        Output times, shape ``(n_t,)`` -- the requested ``t_eval`` grid, or, when
        ``t_eval`` is None, the time the solve ended (``t1``, or the firing time
        of a terminal event that stopped it early).
    ys : jnp.ndarray
        States at ``ts``, shape ``(n_t, n_state)``. A reported time that
        coincides with an event firing carries the **pre-reset** (left-limit)
        state, whether it comes from ``t_eval`` or is the final time reported
        when ``t_eval`` is None.
    log : list[tuple[float, str]]
        The fired events, in order, as ``(time, name)`` -- the audit trail of
        switch times.
    """

    ts: jnp.ndarray
    ys: jnp.ndarray
    log: list = field(default_factory=list)


@dataclass(frozen=True, eq=False)
class SegmentSettings:
    """The integrator settings every segment of an evented solve shares.

    Bundles what :func:`build_segment_solver` bakes into the compiled
    one-segment solve besides the right-hand side and the state events, so the
    builder's signature stays short. It holds an array (``atol``), so it is not
    itself hashable; :meth:`cache_key` gives the hashable key over its values.

    Attributes
    ----------
    rtol : float
        Relative step tolerance.
    atol : float or jnp.ndarray
        Absolute step tolerance, scalar or shape ``(n_state,)``.
    max_steps : int
        Step budget of each segment.
    dtmax : float or None
        Step cap (native time), ``None`` for uncapped.
    adjoint : diffrax.AbstractAdjoint or None
        The diffrax adjoint, ``None`` for the default.
    order : int
        ESDIRK order of the canonical solver (3 or 5).
    factormax : float or None
        Cap on the PID step growth, ``None`` for the diffrax default.
    solver : diffrax.AbstractSolver or None
        An explicit solver override, ``None`` for the canonical one.
    root_rtol, root_atol : float
        Tolerances of the root find that locates a state event.
    max_segments : int
        Safety cap on the number of state-event segments (a runaway-event
        guard). Read by the eager driver, not compiled.
    """

    rtol: float
    atol: object
    max_steps: int = 100_000
    dtmax: float | None = None
    adjoint: diffrax.AbstractAdjoint | None = None
    order: int = 5
    factormax: float | None = None
    solver: object = None
    root_rtol: float = 1e-6
    root_atol: float = 1e-9
    max_segments: int = 10_000

    def cache_key(self):
        """Hashable key over every setting, or ``None`` when one is traced.

        The solver settings go through
        :func:`~aquakin.integrate._cache.concrete_settings_key` (the field order
        every compiled-solve cache shares), followed by the root-find
        tolerances. ``None`` (a traced ``atol``, under an outer trace) tells the
        caller to build uncached.
        """
        key = concrete_settings_key(
            self.rtol,
            self.atol,
            self.adjoint,
            self.dtmax,
            self.max_steps,
            order=self.order,
            factormax=self.factormax,
            solver=self.solver,
        )
        return None if key is None else (key, float(self.root_rtol), float(self.root_atol))


def root_events_of(events: Sequence[Event]) -> list[Event]:
    """The state (root-crossing) events of ``events``, in order."""
    return [ev for ev in events if not ev.is_time_event]


def segment_signature(events: Sequence[Event]) -> tuple:
    """Hashable description of the state events a compiled segment locates.

    The compiled one-segment solve built by :func:`build_segment_solver`
    depends on the events only through the state events' ``cond_fn`` and
    ``direction`` (in order: their position indexes the fired mask). Time
    events, ``apply`` resets, ``terminal`` flags and names are handled by the
    eager driver between segments and do not enter the compiled program. Each
    ``cond_fn`` is keyed by its own equality, which for a plain function is its
    identity, so reusing the same :class:`Event` objects reuses the compile.

    Parameters
    ----------
    events : sequence of Event
        The events of the solve.

    Returns
    -------
    tuple
        ``((cond_fn, direction), ...)`` over the state events; ``()`` when
        there are only time events.
    """
    return tuple((ev.cond_fn, ev.direction) for ev in root_events_of(events))


def build_segment_solver(
    make_rhs: Callable,
    settings: SegmentSettings,
    root_events: Sequence[Event],
    *,
    has_t_eval: bool,
):
    """Build the jit-compiled one-segment solve of the event driver.

    Each segment is integrated by the canonical :func:`_run_diffeqsolve` -- the
    **same** integrator (``build_implicit_solver`` + ``PIDController`` +
    adjoint) the plain reactor / plant solves use -- adding only the
    per-segment ``saveat`` and the terminating ``diffrax.Event`` of the state
    events. The segment bounds, the save times, the initial state, the
    parameters and the right-hand-side context are all **runtime arguments**,
    so one compiled program serves every segment of a solve and every repeat
    solve: a caller that keeps the returned function (keyed by the settings,
    ``has_t_eval`` and :func:`segment_signature`) pays the compile once.

    Parameters
    ----------
    make_rhs : callable
        ``make_rhs(ctx) -> rhs`` building ``rhs(t, y, args)`` from the runtime
        context ``ctx`` (a pytree; e.g. the plant's cached recycle maps). It is
        called inside the trace, so it must be traceable.
    settings : SegmentSettings
        The integrator settings shared by every segment.
    root_events : sequence of Event
        The state events, located by a terminating ``diffrax.Event``.
    has_t_eval : bool
        Whether the solve reports a ``t_eval`` grid (the segment then keeps its
        dense output and evaluates it on the grid).

    Returns
    -------
    callable
        ``solve_segment(ctx, y0, t0, t1, args, t_eval, tol)`` returning
        ``((t_end, y_end, fired, mask, ys), result)``, the solve's diffrax
        ``result`` for :func:`~aquakin.integrate._solver.check_solve_result`
        beside the outputs: where the segment stopped and its
        state there, whether a state event stopped it, the per-state-event
        fired flags, and -- with ``has_t_eval`` -- the state at every ``t_eval``
        point (meaningful for the points in ``(t0, t_end]``; a point within
        ``tol`` of ``t_end`` reports the exact end state), else ``None``.
    """
    diffrax_event = None
    if root_events:
        diffrax_event = diffrax.Event(
            [_wrap_cond(ev) for ev in root_events],
            root_finder=optx.Newton(rtol=settings.root_rtol, atol=settings.root_atol),
            direction=[
                None if ev.direction == 0 else bool(ev.direction == 1) for ev in root_events
            ],
        )

    @jax.jit
    def solve_segment(ctx, y0, t0, t1, args, t_eval, tol):
        sol = _run_diffeqsolve(
            make_rhs(ctx),
            t0=t0,
            t1=t1,
            y0=y0,
            args=args,
            saveat=diffrax.SaveAt(t1=True, dense=has_t_eval),
            rtol=settings.rtol,
            atol=settings.atol,
            adjoint=settings.adjoint,
            max_steps=settings.max_steps,
            dtmax=settings.dtmax,
            event=diffrax_event,
            order=settings.order,
            factormax=settings.factormax,
            solver=settings.solver,
        )
        t_end = sol.ts[-1]
        y_end = sol.ys[-1]
        fired = sol.result == diffrax.RESULTS.event_occurred
        mask = () if diffrax_event is None else tuple(sol.event_mask)
        ys = None
        if has_t_eval:
            # A point at the segment end takes the exact end state rather than
            # the dense interpolant (evaluating diffrax dense output exactly at
            # the right boundary is an edge case that can return NaN, and the
            # end state is the documented pre-reset boundary value). Points
            # outside the segment are evaluated at its midpoint, a safe interior
            # time; the driver discards them.
            at_end = t_eval >= t_end - tol
            inside = (t_eval > t0) & ~at_end
            t_safe = jnp.where(inside, t_eval, 0.5 * (t0 + t_end))
            ys = jax.vmap(sol.evaluate)(t_safe)
            ys = jnp.where(at_end[:, None], y_end, ys)
        return (t_end, y_end, fired, mask, ys), sol.result

    return solve_segment


def _apply_resets(events_to_fire, t, y, args):
    """Apply the ``apply`` resets of the fired events in order; return new ``y``
    and whether any was terminal."""
    terminal = False
    for ev in events_to_fire:
        if ev.apply is not None:
            y = jnp.asarray(ev.apply(t, y, args))
        terminal = terminal or ev.terminal
    return y, terminal


def solve_with_events(
    rhs: Callable,
    y0: jnp.ndarray,
    args,
    *,
    t0: float,
    t1: float,
    t_eval: jnp.ndarray | None,
    events: Sequence[Event],
    rtol: float,
    atol,
    max_steps: int = 100_000,
    dtmax: float | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    order: int = 5,
    factormax: float | None = None,
    solver=None,
    root_rtol: float = 1e-6,
    root_atol: float = 1e-9,
    max_segments: int = 10_000,
) -> EventedResult:
    """Integrate ``rhs`` from ``t0`` to ``t1`` with located events + state resets.

    The solve is split into segments at the event times; between segments the
    fired events' ``apply`` resets produce the new state. With **only time
    events** the segment boundaries are static and every sub-solve is
    differentiable, so ``jax.grad`` flows through the whole call, and the call
    can be wrapped in ``jax.jit`` (``t_eval`` may then be traced). With any
    **state event** the driver runs eagerly (the firing times are discovered at
    runtime) and is forward-simulation only.

    Every segment runs one compiled solve, built once per call; the plant and
    batch-reactor ``events=`` paths keep it across calls as well.

    Parameters
    ----------
    rhs : callable
        ``rhs(t, y, args) -> dy/dt``.
    y0 : jnp.ndarray
        Initial state, shape ``(n_state,)``.
    args
        Parameter argument threaded to ``rhs``, the event ``cond_fn`` / ``apply``.
    t0, t1 : float
        Integration interval.
    t_eval : jnp.ndarray, optional
        Output times (must be sorted, within ``[t0, t1]``). ``None`` returns only
        the final state.
    events : sequence of Event
        The events to locate.
    rtol, atol, max_steps, dtmax, adjoint, order, factormax, solver
        Integrator settings (as for the plain solve), threaded into every
        segment so the event path's integration matches a plain solve exactly.
    root_rtol, root_atol : float
        Tolerances of the root find that locates a state event.
    max_segments : int
        Safety cap on the number of state-event segments (a runaway-event guard).

    Returns
    -------
    EventedResult
    """
    settings = SegmentSettings(
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        dtmax=dtmax,
        adjoint=adjoint,
        order=order,
        factormax=factormax,
        solver=solver,
        root_rtol=root_rtol,
        root_atol=root_atol,
        max_segments=max_segments,
    )
    solve_segment = build_segment_solver(
        lambda ctx: rhs, settings, root_events_of(events), has_t_eval=t_eval is not None
    )
    return run_segmented(
        solve_segment,
        None,
        y0,
        args,
        t_span=(t0, t1),
        t_eval=t_eval,
        events=events,
        settings=settings,
    )


def run_segmented(
    solve_segment,
    ctx,
    y0,
    args,
    *,
    t_span,
    t_eval,
    events,
    settings,
) -> EventedResult:
    """Run the segmented event solve with a (possibly cached) segment solver.

    The shared back half of :func:`solve_with_events` and the plant /
    batch-reactor ``events=`` paths: it labels unnamed events, checks
    ``t_eval``, and drives the segments, translating a segment that exhausts
    its step budget into the friendly remedy message.

    Parameters
    ----------
    solve_segment : callable
        The one-segment solve from :func:`build_segment_solver`, built for the
        state events of ``events`` and for ``t_eval is not None``.
    ctx
        Runtime right-hand-side context passed to ``solve_segment``.
    y0 : jnp.ndarray
        Initial state, shape ``(n_state,)``.
    args
        Parameter argument threaded to the RHS and the events.
    t_span : (float, float)
        Integration interval ``(t0, t1)``.
    t_eval : jnp.ndarray or None
        Sorted output times, or ``None`` for the final state only. May be traced.
    events : sequence of Event
        The events to locate.
    settings : SegmentSettings
        The settings ``solve_segment`` was built with; the driver reads its
        ``max_steps`` (named in the remedy message) and ``max_segments``.

    Returns
    -------
    EventedResult
    """
    events = list(events)
    if not events:
        raise ValueError("solve_with_events needs at least one Event.")
    # Label unnamed events on copies so the caller's Event objects are untouched.
    events = [
        ev if ev.name is not None else dataclasses.replace(ev, name=f"event{i}")
        for i, ev in enumerate(events)
    ]
    if t_eval is not None:
        t_eval = jnp.asarray(t_eval, dtype=float)
        # A traced grid (the solve under an outer jit) cannot be inspected.
        if not isinstance(t_eval, jax.core.Tracer) and np.any(np.diff(np.asarray(t_eval)) < 0):
            raise ValueError("t_eval must be sorted (ascending).")

    # Wrap the segmented drive so a segment exhausting the integrator step budget
    # re-raises the friendly remedy message instead of the raw equinox
    # MaxStepsReached chatter.
    t0, t1 = float(t_span[0]), float(t_span[1])
    with friendly_solve_errors(settings.max_steps, what="event solve"):
        return _drive(solve_segment, ctx, y0, args, t0, t1, t_eval, events, settings)


def _drive(solve_segment, ctx, y0, args, t0, t1, t_eval, events, settings):
    """Segmented event driver, shared by the time- and state-event cases.

    With **no** state event the segment boundaries are the static time-event
    schedule, no terminating ``diffrax.Event`` is used, and nothing branches on
    traced state -- so the whole drive is a fixed sequence of differentiable
    sub-solves: ``jax.grad`` flows through it and it traces under ``jax.jit``.
    With a state event the firing time is located by a root find and discovered
    at runtime, so the loop is eager (forward simulation).

    Output convention: a ``t_eval`` point that coincides with an event time is
    emitted with its **pre-reset** (left-limit) value -- it belongs to the
    segment ending at the event -- so both paths agree on the boundary value and
    the reset defines the next segment's initial condition. A small time
    tolerance makes that assignment robust to the root finder's float error. The
    final state reported when ``t_eval`` is None follows the same convention (an
    event at exactly ``t1``, or a terminal event, is applied and logged, but the
    reported state is its pre-reset value), so ``t_eval=None`` and
    ``t_eval=[t_end]`` agree. A time event at exactly ``t0`` is applied to the
    initial state before the first segment (a ``t_eval`` point at ``t0`` still
    reports the pre-reset initial state).

    The ``t_eval`` points are assigned to segments by mask (each point to the
    first segment whose end it does not exceed, by more than the tolerance), so
    the grid need not be concrete. Each segment's solve result is checked with
    ``settings.max_steps``; ``settings.max_segments`` caps a state-event loop.
    """
    root_events = root_events_of(events)
    has_root = bool(root_events)
    tol = _BOUNDARY_ASSIGN_RTOL * max(1.0, abs(t1 - t0))
    time_times = sorted({t for ev in events for t in (ev.at_times or []) if t0 + tol < t < t1})

    log = []
    y = jnp.asarray(y0)
    out = claimed = None
    if t_eval is not None:
        # Points at the very start (t == t0) take the initial state.
        claimed = t_eval <= t0 + tol
        out = jnp.where(claimed[:, None], y, jnp.zeros((t_eval.shape[0], y.shape[0]), y.dtype))
    seg_t0 = t0
    # The reported final time/state when t_eval is None: the end of the last
    # segment, before any reset applied there (the pre-reset convention).
    t_report, y_report = t0, y

    # Time events at exactly t0 act on the initial state before integration.
    firing = [ev for ev in events if ev.at_times and any(abs(t - t0) <= tol for t in ev.at_times)]
    y, terminal = _apply_resets(firing, t0, y, args)
    for ev in firing:
        log.append((t0, ev.name))
    if terminal:
        return _assemble(out, claimed, t_eval, y, t_report, y_report, log)

    n_iter = settings.max_segments if has_root else len(time_times) + 1
    for _ in range(n_iter):
        if seg_t0 >= t1:
            break
        next_time = next((t for t in time_times if t > seg_t0 + tol), t1)
        seg_t1 = min(next_time, t1)
        if seg_t1 <= seg_t0:
            seg_t0 = seg_t1
            continue
        segment, result = solve_segment(ctx, y, seg_t0, seg_t1, args, t_eval, tol)
        t_seg_end, y_end, fired, mask, ys = check_solve_result(
            result, segment, max_steps=settings.max_steps, what="event solve"
        )
        if has_root:
            fired_root = bool(fired)
            t_end = float(t_seg_end)
        else:
            fired_root = False
            t_end = seg_t1
        t_report, y_report = t_end, y_end

        # Emit the t_eval points in (previous end, t_end] (pre-reset).
        if t_eval is not None:
            own = ~claimed & (t_eval <= t_end + tol)
            out = jnp.where(own[:, None], ys, out)
            claimed = claimed | own

        # Determine which events fire at the segment end and apply their resets.
        if fired_root:
            fired_mask = [bool(m) for m in mask]
            firing = [root_events[i] for i in range(len(root_events)) if fired_mask[i]]
        elif not has_root or t_end >= seg_t1 - tol:
            firing = [
                ev
                for ev in events
                if ev.at_times and any(abs(seg_t1 - t) <= tol for t in ev.at_times)
            ]
        else:
            firing = []
        y, terminal = _apply_resets(firing, t_end, y_end, args)
        for ev in firing:
            log.append((t_end, ev.name))
        seg_t0 = t_end
        if terminal:
            break
    else:
        if has_root:
            raise RuntimeError(
                f"event solve exceeded max_segments={settings.max_segments}; a state "
                f"event may be firing repeatedly without advancing (check its "
                f"apply reset moves the state off the threshold)."
            )

    return _assemble(out, claimed, t_eval, y, t_report, y_report, log)


@dataclass(frozen=True)
class _InRequestedTime:
    """An event callable that receives ``t`` in the solve's requested unit.

    ``fn(t / factor, y, args)`` for a native-time ``t``. A value object rather
    than a lambda so two conversions of the same ``fn`` compare equal, which
    lets a compiled segment keyed on a state event's ``cond_fn``
    (:func:`segment_signature`) be reused across solves.
    """

    fn: Callable
    factor: float

    def __call__(self, t, y, args):
        return self.fn(t / self.factor, y, args)


def events_to_native_time(events: Sequence[Event], time_factor: float) -> list[Event]:
    """Convert events from a solve's requested time unit to the native unit.

    Returns copies (the caller's :class:`Event` objects are not modified) whose
    ``at_times`` are scaled into native time, and whose ``cond_fn`` / ``apply``
    are wrapped so they still receive ``t`` in the requested unit.

    Parameters
    ----------
    events : sequence of Event
        Events expressed in the requested time unit.
    time_factor : float
        Native time units per requested time unit (``t_native = t * factor``),
        as returned by :func:`aquakin.integrate._common.to_native_time`.

    Returns
    -------
    list of Event
        The converted copies (the input list itself when ``time_factor == 1``).
    """
    if time_factor == 1.0:
        return list(events)

    def rescale(fn):
        return None if fn is None else _InRequestedTime(fn, float(time_factor))

    return [
        dataclasses.replace(
            ev,
            at_times=None if ev.at_times is None else [t * time_factor for t in ev.at_times],
            cond_fn=rescale(ev.cond_fn),
            apply=rescale(ev.apply),
        )
        for ev in events
    ]


def _wrap_cond(ev: Event):
    """Adapt our ``cond_fn(t, y, args)`` to diffrax's ``(t, y, args, **kwargs)``."""
    fn = ev.cond_fn
    return lambda t, y, args, **kwargs: fn(t, y, args)


def _assemble(out, claimed, t_eval, y_final, t_end, y_end, log):
    """Build the final EventedResult on the requested ``t_eval`` grid.

    ``out`` holds the states emitted on ``t_eval`` so far, ``claimed`` which of
    its rows are filled. A terminal event that stopped the solve early leaves
    the unreached ``t_eval`` points unfilled: they take the final post-event
    state ``y_final`` (the solve holds there after terminating), so ``ys``
    always matches the ``t_eval`` shape. With ``t_eval`` None the result is the
    single point ``(t_end, y_end)``: the time the solve actually stopped and its
    pre-reset state there.
    """
    if t_eval is None:
        return EventedResult(ts=jnp.asarray([t_end]), ys=y_end[None, :], log=log)
    ys = jnp.where(claimed[:, None], out, y_final)
    return EventedResult(ts=t_eval, ys=ys, log=log)
