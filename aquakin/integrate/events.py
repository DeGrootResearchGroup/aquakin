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

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import diffrax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from aquakin.integrate._common import _run_diffeqsolve


@dataclass
class Event:
    """A located discontinuity in a solve: a time or state trigger + a reset.

    Exactly one trigger must be given:

    * ``at_times`` -- a sequence of times (in the solve's time unit) at which the
      event fires. AD-safe: the segment boundaries are static.
    * ``cond_fn`` -- a scalar function ``cond_fn(t, y, args)`` whose zero crossing
      (in ``direction``) is the event. Located by a root find; forward-simulation
      only (the firing count is data-dependent).

    Parameters
    ----------
    cond_fn : callable, optional
        Root-crossing condition ``cond_fn(t, y, args) -> scalar``. ``y`` is the
        full state vector (the reactor concentration vector, or the flat plant
        state); ``args`` is the solve's parameter argument.
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

    cond_fn: Optional[Callable] = None
    at_times: Optional[Sequence[float]] = None
    direction: int = 0
    apply: Optional[Callable] = None
    terminal: bool = False
    name: Optional[str] = None

    def __post_init__(self):
        has_cond = self.cond_fn is not None
        has_times = self.at_times is not None
        if has_cond == has_times:
            raise ValueError(
                "an Event needs exactly one trigger: cond_fn (state event) OR "
                "at_times (time event).")
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
        Output times, shape ``(n_t,)`` -- the requested ``t_eval`` grid (or the
        final time when ``t_eval`` is None).
    ys : jnp.ndarray
        States at ``ts``, shape ``(n_t, n_state)``.
    log : list[tuple[float, str]]
        The fired events, in order, as ``(time, name)`` -- the audit trail of
        switch times.
    """

    ts: jnp.ndarray
    ys: jnp.ndarray
    log: list = field(default_factory=list)


def _make_segment_solver(rhs, *, rtol, atol, max_steps, dtmax, adjoint):
    """Build a one-segment solver for the event driver.

    Delegates to the canonical :func:`_run_diffeqsolve` -- the **same**
    Kvaerno5 + ``PIDController`` + adjoint setup the plain reactor/plant solves
    use -- adding only the per-segment ``saveat`` and terminating ``event``. So
    the event path's per-step integration is the plain path's, and the solver
    defaults cannot drift between them.
    """
    def solve_segment(y0, t0, t1, args, saveat, event):
        return _run_diffeqsolve(
            rhs, t0=t0, t1=t1, y0=y0, args=args, saveat=saveat,
            rtol=rtol, atol=atol, adjoint=adjoint, max_steps=max_steps,
            dtmax=dtmax, event=event,
        )

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
    t_eval: Optional[jnp.ndarray],
    events: Sequence[Event],
    rtol: float,
    atol,
    max_steps: int = 100_000,
    dtmax: Optional[float] = None,
    adjoint: Optional[diffrax.AbstractAdjoint] = None,
    root_rtol: float = 1e-6,
    root_atol: float = 1e-9,
    max_segments: int = 10_000,
) -> EventedResult:
    """Integrate ``rhs`` from ``t0`` to ``t1`` with located events + state resets.

    The solve is split into segments at the event times; between segments the
    fired events' ``apply`` resets produce the new state. With **only time
    events** the segment boundaries are static and every sub-solve is
    differentiable, so ``jax.grad`` flows through the whole call. With any
    **state event** the driver runs eagerly (the firing times are discovered at
    runtime) and is forward-simulation only.

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
    rtol, atol, max_steps, dtmax, adjoint
        Solver settings (as for the plain solve).
    root_rtol, root_atol : float
        Tolerances of the root find that locates a state event.
    max_segments : int
        Safety cap on the number of state-event segments (a runaway-event guard).

    Returns
    -------
    EventedResult
    """
    events = list(events)
    if not events:
        raise ValueError("solve_with_events needs at least one Event.")
    for i, ev in enumerate(events):
        if ev.name is None:
            ev.name = f"event{i}"

    t0 = float(t0)
    t1 = float(t1)
    t_eval_np = None if t_eval is None else np.asarray(t_eval, dtype=float)
    if t_eval_np is not None and np.any(np.diff(t_eval_np) < 0):
        raise ValueError("t_eval must be sorted (ascending).")

    solve_segment = _make_segment_solver(
        rhs, rtol=rtol, atol=atol, max_steps=max_steps, dtmax=dtmax,
        adjoint=adjoint)

    has_root = bool([ev for ev in events if not ev.is_time_event])
    return _drive(solve_segment, y0, args, t0, t1, t_eval_np, events,
                  has_root, root_rtol, root_atol, max_segments)


def _drive(solve_segment, y0, args, t0, t1, t_eval_np, events,
           has_root, root_rtol, root_atol, max_segments):
    """Segmented event driver, shared by the time- and state-event cases.

    With **no** state event (``has_root=False``) the segment boundaries are the
    static time-event schedule, no terminating ``diffrax.Event`` is used, and
    nothing branches on traced state -- so the whole drive is a fixed sequence of
    differentiable sub-solves and ``jax.grad`` flows through it. With a state
    event the firing time is located by a root find and discovered at runtime, so
    the loop is eager (forward simulation).

    Output convention: a ``t_eval`` point that coincides with an event time is
    emitted with its **pre-reset** (left-limit) value -- it belongs to the
    segment ending at the event -- so both paths agree on the boundary value and
    the reset defines the next segment's initial condition. A small time
    tolerance makes that assignment robust to the root finder's float error.
    """
    root_events = [ev for ev in events if not ev.is_time_event]
    time_times = sorted({t for ev in events
                         for t in (ev.at_times or []) if t0 < t < t1})
    diffrax_event = None
    if has_root:
        root = optx.Newton(rtol=root_rtol, atol=root_atol)
        diffrax_event = diffrax.Event(
            [_wrap_cond(ev) for ev in root_events], root_finder=root,
            direction=[None if ev.direction == 0 else bool(ev.direction == 1)
                       for ev in root_events])
    tol = 1e-9 * max(1.0, abs(t1 - t0))   # boundary-assignment tolerance

    log = []
    out_y = []
    idx = 0                                # cursor into t_eval
    n_eval = 0 if t_eval_np is None else t_eval_np.shape[0]
    y = jnp.asarray(y0)
    seg_t0 = t0
    # Emit any t_eval points at the very start (t == t0) with the initial state.
    while idx < n_eval and t_eval_np[idx] <= t0 + tol:
        out_y.append(y)
        idx += 1

    n_iter = max_segments if has_root else len(time_times) + 1
    for _ in range(n_iter):
        if seg_t0 >= t1:
            break
        next_time = next((t for t in time_times if t > seg_t0 + tol), t1)
        seg_t1 = min(next_time, t1)
        if seg_t1 <= seg_t0:
            seg_t0 = seg_t1
            continue
        saveat = diffrax.SaveAt(t1=True, dense=(n_eval > 0))
        sol = solve_segment(y, seg_t0, seg_t1, args, saveat, diffrax_event)
        if has_root:
            fired_root = bool(sol.result == diffrax.RESULTS.event_occurred)
            t_end = float(sol.ts[-1])
        else:
            fired_root = False
            t_end = seg_t1
        y_end = sol.ys[-1]

        # Emit t_eval points in (seg_t0, t_end] (pre-reset). A point coinciding
        # with the segment end uses the exact endpoint state ``y_end`` rather than
        # the dense interpolant: evaluating diffrax dense output exactly at the
        # right boundary t1 is an edge case that can return NaN, and the endpoint
        # is the value we want anyway (the documented pre-reset boundary value).
        while idx < n_eval and t_eval_np[idx] <= t_end + tol:
            te = float(t_eval_np[idx])
            out_y.append(y_end if te >= t_end - tol else sol.evaluate(te))
            idx += 1

        # Determine which events fire at the segment end and apply their resets.
        if fired_root:
            mask = [bool(m) for m in sol.event_mask]
            firing = [root_events[i] for i in range(len(root_events)) if mask[i]]
        elif not has_root or t_end >= seg_t1 - tol:
            firing = [ev for ev in events
                      if ev.at_times and any(abs(seg_t1 - t) <= tol
                                             for t in ev.at_times)]
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
                f"event solve exceeded max_segments={max_segments}; a state "
                f"event may be firing repeatedly without advancing (check its "
                f"apply reset moves the state off the threshold).")

    return _assemble(out_y, t_eval_np, y, t1, log)


def _wrap_cond(ev: Event):
    """Adapt our ``cond_fn(t, y, args)`` to diffrax's ``(t, y, args, **kwargs)``."""
    fn = ev.cond_fn
    return lambda t, y, args, **kwargs: fn(t, y, args)


def _assemble(out_y, t_eval_np, y_final, t1, log):
    """Build the final EventedResult on the requested ``t_eval`` grid.

    ``out_y`` holds the states emitted on ``t_eval`` so far (in grid order). A
    terminal event that stopped the solve early leaves the unreached ``t_eval``
    points unfilled: pad them with the final post-event state (the solve holds
    there after terminating), so ``ys`` always matches the ``t_eval`` shape.
    """
    if t_eval_np is None:
        return EventedResult(ts=jnp.asarray([t1]), ys=y_final[None, :], log=log)
    while len(out_y) < t_eval_np.shape[0]:
        out_y.append(y_final)
    return EventedResult(ts=jnp.asarray(t_eval_np), ys=jnp.stack(out_y, axis=0),
                         log=log)
