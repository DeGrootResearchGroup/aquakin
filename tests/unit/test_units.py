"""Unit-string prettifier tests."""

import pytest

from aquakin.core.units import prettify_units


@pytest.mark.parametrize(
    "plain, pretty",
    [
        ("g_COD/m3", "g_COD/m³"),
        ("g_N/m3", "g_N/m³"),
        ("mol/m3", "mol/m³"),
        ("M-1 s-1", "M⁻¹ s⁻¹"),
        ("1/d", "1/d"),          # leading literal 1 is not a unit symbol
        ("mol/L", "mol/L"),      # no exponents
        ("-", "-"),             # dimensionless
        ("", ""),               # empty
        ("m2", "m²"),
        ("d-1", "d⁻¹"),
        # Longest-match-first alternation: `min` / `mol` must win over `m`, so the
        # whole symbol carries the exponent (not the `m` leaving `in1` / `ol1`).
        ("min-1", "min⁻¹"),
        ("mol-1", "mol⁻¹"),
        ("mol2", "mol²"),
        # Multi-character symbols carry exponents too.
        ("kg2", "kg²"),
        ("Pa-1", "Pa⁻¹"),
    ],
)
def test_prettify_units(plain, pretty):
    assert prettify_units(plain) == pretty


def test_chemical_subscripts_are_not_exponents():
    """A number that is a chemical subscript (O2, CO2) must be left alone, even
    though a unit exponent in the same string is converted."""
    assert prettify_units("g_O2/m3") == "g_O2/m³"
    assert prettify_units("g_CO2/m3") == "g_CO2/m³"


def test_idempotent():
    once = prettify_units("g_O2/m3")
    assert prettify_units(once) == once
