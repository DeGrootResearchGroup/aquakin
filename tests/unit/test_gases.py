"""The gas library and the ``gases:`` model block (aquakin.core.gases).

The library values are checked against independent sources: BSM2's Henry
constants and water vapour pressure (as written in the shipped ``adm1``
model), Siegrist et al.'s dimensionless digester Henry constants, the measured
CO2/O2 transfer-coefficient ratio of Lizarralde et al., and the carbonate
equilibrium in closed form.
"""

from __future__ import annotations

import math
import textwrap

import pytest

import aquakin
from aquakin.core.gases import (
    GAS_LIBRARY,
    R_GAS,
    DissolvedGas,
    henry,
    kL_ratio,
    partial_pressure,
    saturation,
    volatile_concentration,
    water_vapour_pressure,
)

T35 = 308.15


def _entry(gas, molar_mass=1000.0, volatile_fraction=None):
    props = GAS_LIBRARY[gas]
    return DissolvedGas(
        species=gas,
        index=0,
        gas=gas,
        molar_mass=molar_mass,
        henry=props.henry,
        henry_temperature=props.henry_temperature,
        kL_ratio=kL_ratio(gas),
        volatile_fraction=volatile_fraction,
    )


def test_henry_follows_its_van_t_hoff_form():
    e = _entry("CH4")
    expected = 1.4e-5 * math.exp(1600.0 * (1.0 / T35 - 1.0 / 298.15))
    assert float(henry(e, T35)) == pytest.approx(expected, rel=1e-12)
    assert float(henry(e, 298.15)) == pytest.approx(1.4e-5, rel=1e-12)


@pytest.mark.parametrize(
    ("gas", "bsm2_kmol_per_m3_bar"),
    [("CH4", 1.161903e-3), ("CO2", 2.714669e-2), ("H2", 7.384654e-4)],
)
def test_library_solubility_at_35C_agrees_with_bsm2(gas, bsm2_kmol_per_m3_bar):
    """BSM2 (adm1.yaml) writes K_H at 35 C in kmol m^-3 bar^-1 = 100 mol m^-3 Pa^-1.

    The two sources use slightly different temperature coefficients (BSM2's
    CH4 1713 K against the review's 1600 K), so they agree to a few percent.
    """
    ours = float(henry(_entry(gas), T35))
    assert ours == pytest.approx(bsm2_kmol_per_m3_bar / 100.0, rel=0.03)


def test_dimensionless_methane_henry_agrees_with_siegrist():
    """Siegrist et al. (2002) give H_CH4 = 38 (gas over liquid concentration) at
    35 C, raised 10% for the salts of sludge; fresh water is 38/1.1."""
    e = _entry("CH4", molar_mass=1000.0)
    C_gas = 1.0
    p = partial_pressure(e, T35, C_gas)
    C_liq = saturation(e, T35, p)
    assert C_gas / float(C_liq) == pytest.approx(38.0 / 1.1, rel=0.05)


def test_partial_pressure_is_the_ideal_gas_law():
    e = _entry("CO2", molar_mass=12000.0)  # g C/m3 -> mol/L
    # 12 g C per m3 of gas is 1 mol/m3.
    assert float(partial_pressure(e, T35, 12.0)) == pytest.approx(R_GAS * T35, rel=1e-12)


def test_co2_transfer_ratio_matches_the_measured_ratio():
    """Lizarralde et al. (2015) report kLa_CO2/kLa_O2 of 0.88-0.92 in their
    pilot plant; the CRC diffusivities with penetration theory give 0.888."""
    assert 0.87 < kL_ratio("CO2") < 0.93
    assert kL_ratio("O2") == 1.0


def test_volatile_carbon_is_half_the_total_at_pk1():
    """At pH = pK1 the H2CO3* and HCO3- fractions are equal (CO3 negligible)."""
    e = _entry("CO2", volatile_fraction="carbonate")
    k1 = 10.0**-6.35  # Stumm & Morgan at 25 C, as in core.ph_solver
    v = float(volatile_concentration(e, 1.0, 298.15, -math.log10(k1)))
    assert v == pytest.approx(0.5, abs=1e-4)


def test_volatile_carbon_matches_the_adm1_dissolved_co2_at_35C():
    """adm1.yaml takes dissolved CO2 as S_IC * pH_switch(6.3065) at 35 C."""
    e = _entry("CO2", volatile_fraction="carbonate")
    expected = 1.0 / (1.0 + 10.0 ** (7.0 - 6.3065))
    assert float(volatile_concentration(e, 1.0, T35, 7.0)) == pytest.approx(expected, rel=1e-3)


def test_a_gas_without_volatile_fraction_transfers_whole():
    assert float(volatile_concentration(_entry("CH4"), 3.0, T35, 7.0)) == 3.0


def test_water_vapour_pressure_is_the_bsm2_value():
    assert float(water_vapour_pressure(T35)) == pytest.approx(0.0556677e5, rel=1e-6)


# --- the gases: block ----------------------------------------------------------

_MODEL = """
model: {{name: gas_block_test}}
species:
  - {{name: SCH4, units: g_COD/m3}}
  - {{name: SIC, units: g_C/m3}}
conditions:
  - {{name: T, default: 298.15}}
{extra_conditions}
parameters:
  k: {{value: 1.0, units: 1/d}}
gases:
{gases}
reactions:
  - name: r
    rate: "k * [SCH4]"
    stoichiometry: {{SCH4: -1, SIC: 1}}
"""


def _load(tmp_path, gases, extra_conditions=""):
    path = tmp_path / "m.yaml"
    path.write_text(
        _MODEL.format(gases=textwrap.indent(gases, "  "), extra_conditions=extra_conditions)
    )
    return aquakin.load_model_from_file(path)


def test_block_compiles_library_values_and_overrides(tmp_path):
    m = _load(
        tmp_path,
        "pH_field: pH\nspecies:\n"
        "  SCH4: {gas: CH4, molar_mass: 64000, henry: 1.0e-5, kL_ratio: 0.5}\n"
        "  SIC: {gas: CO2, molar_mass: 12000, volatile_fraction: carbonate}\n",
        extra_conditions="  - {name: pH, default: 7.0}",
    )
    ch4 = m.gases.by_gas("CH4")
    co2 = m.gases.by_gas("CO2")
    assert (ch4.index, ch4.henry, ch4.kL_ratio) == (0, 1.0e-5, 0.5)
    assert ch4.henry_temperature == GAS_LIBRARY["CH4"].henry_temperature
    assert (co2.index, co2.henry, co2.kL_ratio) == (1, 3.4e-4, kL_ratio("CO2"))
    assert m.gases.gases == ["CH4", "CO2"]
    with pytest.raises(KeyError, match="declares no dissolved O2"):
        m.gases.by_gas("O2")


def test_celsius_temperature_is_converted(tmp_path):
    m = _load(
        tmp_path,
        "temperature_units: celsius\nspecies:\n  SCH4: {gas: CH4, molar_mass: 64000}\n",
    )
    assert m.gases.kelvin(25.0) == pytest.approx(298.15)


@pytest.mark.parametrize(
    ("gases", "message"),
    [
        ("species:\n  SCH4: {gas: CH5, molar_mass: 64000}\n", "unknown gas 'CH5'"),
        (
            "species:\n  SIC: {gas: CO2, molar_mass: 12000, volatile_fraction: sulfide}\n",
            "volatile_fraction must be one of",
        ),
        ("temperature_units: rankine\nspecies:\n  SCH4: {gas: CH4, molar_mass: 1}\n", "rankine"),
    ],
)
def test_block_rejects_invalid_entries(tmp_path, gases, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, gases)


@pytest.mark.parametrize(
    ("gases", "message"),
    [
        ("species:\n  SX: {gas: CH4, molar_mass: 1}\n", "'SX' is not a species"),
        (
            "species:\n  SCH4: {gas: CH4, molar_mass: 1}\n  SIC: {gas: CH4, molar_mass: 1}\n",
            "one species only",
        ),
        (
            "species:\n  SIC: {gas: CO2, molar_mass: 12000, volatile_fraction: carbonate}\n",
            "needs a pH",
        ),
        ("temperature_field: Tw\nspecies:\n  SCH4: {gas: CH4, molar_mass: 1}\n", "'Tw'"),
    ],
)
def test_block_rejects_inconsistent_models(tmp_path, gases, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, gases)


def test_a_model_without_the_block_has_no_gases():
    assert aquakin.load_model("asm1").gases is None
