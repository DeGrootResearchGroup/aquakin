"""The control-signal bus: checks of who publishes and who reads each signal.

Controllers publish named scalar signals and actuated units read them (see
:mod:`aquakin.plant.control`). The checks here run when a plant's state layout
is built, before the right-hand side is traced, so a wiring mistake is a clear
error naming the unit rather than a ``KeyError`` from inside a jitted solve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from aquakin.plant.units import SignalCheckedUnit, SignalProducer


def validate_control_signals(units: Mapping) -> None:
    """Cross-check the control-signal bus: every published signal name is
    unique, and every signal a unit *consumes* is *published* by some unit.

    A unit under closed-loop control (a ``CSTRUnit`` whose ``aeration`` has a
    DO setpoint) reads ``signals[name]`` in its ``rhs``; if no controller publishes
    ``name`` -- a forgotten or mistyped wiring -- that read is a bare
    ``KeyError`` from deep inside the first jitted solve. This runs at
    topology setup (before the RHS is traced) and raises a clear error
    naming the unit, the missing signal, and the available signals instead.

    Two controllers publishing the *same* signal name are also rejected here:
    the bus is gathered with ``dict.update`` (see :meth:`Plant._compute_signals`),
    so a duplicate would silently overwrite -- whichever unit runs later wins
    and the other's output is discarded while its integral state keeps
    winding. That is caught up front as a clear error rather than a silent
    wrong closed loop.

    Conservative by design: a unit that publishes signals declares their
    names via ``signal_names``; if any signal *producer* (one exposing
    ``signal_outputs``) does not declare ``signal_names``, the published set
    is unknown, so validation is skipped rather than risk rejecting a valid
    plant.

    A :class:`~aquakin.plant.units.SignalCheckedUnit` checks its own reads
    against the published signals first, so its message can be specific.

    Parameters
    ----------
    units : mapping of str to Unit
        The plant's units, by name.

    Raises
    ------
    ValueError
        If two units publish one signal, or a unit reads a signal none publishes.
    """
    unknown_publisher = any(
        hasattr(u, "signal_outputs") and not hasattr(u, "signal_names") for u in units.values()
    )
    if unknown_publisher:
        return
    published: set[str] = set()
    publisher_of: dict[str, str] = {}
    for name, unit in units.items():
        for sig in getattr(unit, "signal_names", ()) or ():
            if sig in publisher_of:
                raise ValueError(
                    f"Control signal '{sig}' is published by both "
                    f"'{publisher_of[sig]}' and '{name}'. Signal names must be "
                    f"unique -- the bus gathers them by name, so a duplicate "
                    f"would silently overwrite. Give the controllers distinct "
                    f"signal names."
                )
            publisher_of[sig] = name
            published.add(sig)
    for unit in units.values():
        if isinstance(unit, SignalCheckedUnit):
            unit.check_signals(dict(publisher_of))
    for name, unit in units.items():
        for sig in getattr(unit, "required_signals", ()) or ():
            if sig not in published:
                raise ValueError(
                    f"Unit '{name}' consumes control signal '{sig}', which "
                    f"no unit in this plant publishes. Published signals: "
                    f"{sorted(published) or '(none)'}. Add a controller "
                    f"(e.g. a PIController) that publishes '{sig}', or fix "
                    f"the signal name."
                )


def order_signal_producers(units: Mapping, unit_order: Sequence[str]) -> tuple[str, ...]:
    """The signal producers, each after the producers whose signals it reads.

    A producer reads the bus built before it (the cascade of one law reading
    another's output), so the plant gathers the bus in this order. It is the
    plant's unit order wherever the reads allow, so a plant without cascades
    computes its signals exactly as before. A producer that declares no
    ``signal_names`` publishes nothing known and is placed by its reads alone.

    Parameters
    ----------
    units : mapping of str to Unit
        The plant's units, by name.
    unit_order : sequence of str
        The plant's unit order, which breaks ties.

    Returns
    -------
    tuple of str
        The names of the :class:`~aquakin.plant.units.SignalProducer` units.

    Raises
    ------
    ValueError
        If producers read each other's signals in a cycle; the message names it.
    """
    producers = [n for n in unit_order if isinstance(units[n], SignalProducer)]
    publisher = {sig: n for n in producers for sig in getattr(units[n], "signal_names", ()) or ()}
    reads = {
        n: {
            publisher[sig]: sig
            for sig in getattr(units[n], "required_signals", ()) or ()
            if sig in publisher
        }
        for n in producers
    }
    order: list[str] = []
    pending = list(producers)
    while pending:
        ready = next((n for n in pending if all(d in order for d in reads[n])), None)
        if ready is None:
            raise ValueError(_cycle_message(pending, reads))
        order.append(ready)
        pending.remove(ready)
    return tuple(order)


def _cycle_message(pending: list[str], reads: dict[str, dict[str, str]]) -> str:
    """Name one cycle among the producers left unordered.

    Every one of them reads another of them, so following reads from the first
    must come back to a producer already seen.
    """
    path = [pending[0]]
    while True:
        dep = next(d for d in reads[path[-1]] if d in pending)
        if dep in path:
            cycle = path[path.index(dep) :] + [dep]
            break
        path.append(dep)
    steps = ", ".join(f"'{a}' reads '{reads[a][b]}' from '{b}'" for a, b in zip(cycle, cycle[1:]))
    return (
        f"Signal producers read each other in a cycle: {' -> '.join(cycle)} ({steps}). "
        f"Each producer is computed from the signals published before it, so a cycle "
        f"has no order; break it, for example by reading a measured stream instead."
    )
