"""The plant exception taxonomy and the narrowed digester-gas catch.

Pins the two backward-compatibility-critical properties of ``aquakin.plant.errors``:

- each typed exception subclasses the built-in it historically raised, so
  existing ``except KeyError`` / ``except ValueError`` handlers and message-based
  tests keep working; and
- the "no digester -> no biogas" convenience in the mass balance swallows *only*
  :class:`NoDigesterError`, so a genuine bug inside ``digester_gas`` is no longer
  silently reported as "no biogas".
"""

import pytest

import aquakin
from aquakin.plant.errors import (
    NoDigesterError,
    UnknownPortError,
    UnknownUnitError,
    WiringError,
)


def test_taxonomy_subclasses_preserve_builtin_types():
    # Unknown-name errors are KeyError; wiring/usage errors are ValueError.
    assert issubclass(UnknownUnitError, KeyError)
    assert issubclass(UnknownPortError, KeyError)
    assert issubclass(WiringError, ValueError)
    assert issubclass(NoDigesterError, ValueError)
    # KeyError and ValueError are disjoint, so the two families never overlap.
    assert not issubclass(UnknownUnitError, ValueError)
    assert not issubclass(WiringError, KeyError)


def test_taxonomy_is_exported_at_package_level():
    # Re-exported from both aquakin.plant and the top-level package, and the same
    # objects (not accidental duplicates).
    assert aquakin.UnknownUnitError is UnknownUnitError
    assert aquakin.UnknownPortError is UnknownPortError
    assert aquakin.WiringError is WiringError
    assert aquakin.NoDigesterError is NoDigesterError
    assert aquakin.plant.UnknownUnitError is UnknownUnitError


def test_biogas_cod_swallows_only_no_digester_error(monkeypatch):
    """``_biogas_cod`` returns None for a genuine 'no digester' (its documented
    informational behaviour) but must NOT swallow any other error from
    ``digester_gas`` -- that would hide a real bug as 'no biogas'."""
    from aquakin.plant import balance

    # A NoDigesterError means the plant simply has no digester -> informational
    # None (the biogas term is omitted).
    def _no_digester(*args, **kwargs):
        raise NoDigesterError("this plant has no anaerobic digester")

    monkeypatch.setattr("aquakin.plant.bsm.evaluation.digester_gas", _no_digester)
    assert balance._biogas_cod(object(), object(), None) is None

    # Any other ValueError is a genuine failure inside digester_gas and must
    # propagate, not be relabelled "no biogas".
    def _real_bug(*args, **kwargs):
        raise ValueError("genuine bug inside digester_gas")

    monkeypatch.setattr("aquakin.plant.bsm.evaluation.digester_gas", _real_bug)
    with pytest.raises(ValueError, match="genuine bug inside digester_gas"):
        balance._biogas_cod(object(), object(), None)
