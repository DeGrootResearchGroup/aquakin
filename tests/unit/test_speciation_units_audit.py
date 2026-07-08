"""Advisory audit of speciation/precipitation ``molar_mass`` vs species units.

``molar_mass`` converts a species' state value to mol/L (``mol/L = state /
molar_mass``), so for an already-molar species it must be a pure unit-conversion
factor (a power of ten) and for a mass species a molecular weight (never a clean
power of ten, always well above 1). The relationship lives only in a YAML
comment, so a hand-edit that breaks it silently shifts the computed pH /
saturation index; :class:`aquakin.SpeciationUnitsWarning` flags the likely
mismatch at load time. These tests pin: the dimensional classifier, the
power-of-ten test, that a broken block warns, and -- critically -- that **no
shipped model warns** (zero false positives).
"""

import glob
import os
import warnings

import pytest

import aquakin
from aquakin.schema.model_spec import (
    SpeciationUnitsWarning,
    _is_power_of_ten,
    _units_measure,
)


def _load_capture(tmp_path, body: str):
    """Load a YAML model, returning the SpeciationUnitsWarning messages raised."""
    p = tmp_path / "m.yaml"
    p.write_text(body)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aquakin.load_model_from_file(str(p))
    return [str(w.message) for w in caught if issubclass(w.category, SpeciationUnitsWarning)]


_TEMPLATE = """
model: {{name: t, description: d}}
species:
  - {{name: S_IC, units: {units}, default_concentration: 1.0}}
  - {{name: A, units: mol/L, default_concentration: 1.0}}
conditions:
  - {{name: T, default: 293.15}}
speciation:
  temperature_field: T
  temperature_units: kelvin
  totals:
    carbonate: {{species: S_IC, molar_mass: {mm}}}
reactions:
  - {{name: R, rate: "k*[A]", parameters: {{k: {{value: 0.1}}}}, stoichiometry: {{A: -1}}}}
"""


# --- the dimensional classifier ----------------------------------------------


@pytest.mark.parametrize(
    "units,expected",
    [
        ("mol/L", "molar"),
        ("mmol/L", "molar"),
        ("mol/m3", "molar"),
        ("kmol/m3", "molar"),
        ("kmolC/m3", "molar"),
        ("gC/m3", "mass"),
        ("gCOD/m3", "mass"),
        ("gN/m3", "mass"),
        ("kgCOD/m3", "mass"),
        ("", None),  # blank -> unknown/skip
        ("nonsense-unit", None),  # unparseable -> skip
        ("-", None),  # dimensionless -> neither
    ],
)
def test_units_measure_classification(units, expected):
    assert _units_measure(units) == expected


@pytest.mark.parametrize(
    "value,expected",
    [(1.0, True), (1000.0, True), (0.001, True), (1e6, True),
     (64.0, False), (112.0, False), (12000.0, False), (0.0, False), (-1.0, False)],
)
def test_is_power_of_ten(value, expected):
    assert _is_power_of_ten(value) is expected


# --- load-time advisory behaviour --------------------------------------------


def test_molar_species_with_molecular_weight_warns(tmp_path):
    # S_IC in mol/m3 (already an amount) but molar_mass looks like a MW.
    msgs = _load_capture(tmp_path, _TEMPLATE.format(units="mol/m3", mm="64000"))
    assert any("looks like a molecular weight" in m for m in msgs)


def test_molar_species_with_unit_factor_is_clean(tmp_path):
    # mol/m3 -> mol/L is a factor of 1000 (a power of ten): correct, no warning.
    assert _load_capture(tmp_path, _TEMPLATE.format(units="mol/m3", mm="1000")) == []
    # kmol/m3 == mol/L, factor 1.0.
    assert _load_capture(tmp_path, _TEMPLATE.format(units="kmol/m3", mm="1.0")) == []


def test_mass_species_without_molecular_weight_warns(tmp_path):
    # gC/m3 is a mass, but molar_mass=1.0 applies no molecular weight.
    msgs = _load_capture(tmp_path, _TEMPLATE.format(units="gC/m3", mm="1.0"))
    assert any("below any molecular weight" in m for m in msgs)


def test_mass_species_with_molecular_weight_is_clean(tmp_path):
    # gC/m3 with 12 g/mol * 1000 (m3->L) = 12000: correct, no warning.
    assert _load_capture(tmp_path, _TEMPLATE.format(units="gC/m3", mm="12000")) == []


def test_unparseable_units_are_skipped(tmp_path):
    # An unrecognized unit string cannot be checked, so it is skipped (no noise),
    # mirroring check_units' treatment of unknown units.
    assert _load_capture(tmp_path, _TEMPLATE.format(units='"weird-unit"', mm="5")) == []


def test_warning_is_filterable_by_public_category():
    assert aquakin.SpeciationUnitsWarning is SpeciationUnitsWarning
    assert issubclass(SpeciationUnitsWarning, UserWarning)


# --- zero false positives on shipped models ----------------------------------


def _shipped_speciation_models():
    root = os.path.dirname(aquakin.__file__)
    out = []
    for path in sorted(glob.glob(os.path.join(root, "models", "*.yaml"))):
        text = open(path).read()
        if "speciation:" in text or "precipitation:" in text:
            out.append(path)
    return out


@pytest.mark.parametrize("path", _shipped_speciation_models())
def test_shipped_models_do_not_warn(path):
    """Every shipped model with a speciation/precipitation block must load
    without a SpeciationUnitsWarning -- the check must have zero false positives,
    or the advisory becomes noise on every load."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aquakin.load_model_from_file(path)
    offending = [str(w.message) for w in caught if issubclass(w.category, SpeciationUnitsWarning)]
    assert offending == [], f"{os.path.basename(path)} raised: {offending}"
