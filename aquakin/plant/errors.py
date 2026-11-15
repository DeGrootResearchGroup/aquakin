"""Typed exceptions for plant assembly, wiring, and introspection.

A small, uniform taxonomy so a caller can tell an **unknown name** apart from an
**invalid wiring / unsupported request** -- previously the same condition raised
``KeyError`` in some methods and ``ValueError`` in others, so ``except`` clauses
could not distinguish "bad name" from "bad value".

Each class subclasses the built-in exception that the corresponding sites
historically raised, so existing ``except KeyError`` / ``except ValueError``
handlers and message-matching tests keep working unchanged:

- an **unknown name** (a unit or port that does not exist) -> ``KeyError`` family
  (:class:`UnknownUnitError`, :class:`UnknownPortError`);
- an **invalid wiring or unsupported request** -> ``ValueError`` family
  (:class:`WiringError`, :class:`NoDigesterError`, :class:`UnitsError`).
"""


class UnknownUnitError(KeyError):
    """A referenced unit name is not present in the plant."""


class UnknownPortError(KeyError):
    """A known unit has no input/output port of the requested name."""


class WiringError(ValueError):
    """A structurally invalid connection or an unsupported unit request.

    Raised when a name resolves but its *use* is wrong: an influent used where a
    unit is expected, a bare endpoint that omits a required port, or a unit that
    does not support the requested operation (e.g. ``set_temperature``).
    """


class NoDigesterError(ValueError):
    """The plant has no ADM1 anaerobic digester, so it has no biogas."""


class UnitsError(ValueError):
    """The plant's declared units are inconsistent in a way that makes a solve
    meaningless -- e.g. two kinetic models on different time units (one in
    seconds, one in days) integrated against the same clock. See
    :meth:`Plant.check_units` for the full, non-raising report."""
