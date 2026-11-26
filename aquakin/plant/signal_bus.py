"""The control-signal bus: checks of who publishes and who reads each signal.

Controllers publish named scalar signals and actuated units read them (see
:mod:`aquakin.plant.control`). The checks here run when a plant's state layout
is built, before the right-hand side is traced, so a wiring mistake is a clear
error naming the unit rather than a ``KeyError`` from inside a jitted solve.
"""

from __future__ import annotations


def validate_control_signals(units) -> None:
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
