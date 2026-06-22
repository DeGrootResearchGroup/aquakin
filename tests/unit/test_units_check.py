"""Dimensional ('unit') consistency checking for rate expressions (issue #161).

Three layers: the currency-token :class:`Dimension` algebra, the tolerant
unit-string parser, and the propagation/check over a rate AST. The check must
(a) stay silent on the correctly-annotated shipped networks (no false alarm) and
(b) actually catch the authoring bugs it targets -- a dropped concentration
factor, a Monod term mixing two currencies, a cross-currency sum.

(Distinct from ``test_units.py``, which covers the display prettifier in
``aquakin.core.units``.)
"""
from fractions import Fraction

import pytest

import aquakin
from aquakin.core.parser import parse_rate_expression
from aquakin.utils.units import (
    DIMENSIONLESS,
    Dimension,
    check_rate_units,
    parse_units,
)


# --- the Dimension algebra --------------------------------------------------

def test_dimension_algebra():
    g = Dimension({"g": 1})
    cod = Dimension({"COD": 1})
    conc = g * cod / (Dimension({"m": 1}) ** 3)        # g_COD / m^3
    assert conc == parse_units("g_COD/m3")
    assert (conc / conc).is_dimensionless
    assert DIMENSIONLESS.is_dimensionless
    assert g * g == Dimension({"g": 2})
    assert Dimension({"m": 1}) ** Fraction(1, 2) == Dimension({"m": Fraction(1, 2)})
    assert g / g == DIMENSIONLESS          # zero exponents are dropped


def test_dimension_hashable_and_str():
    assert len({parse_units("g_COD/m3"), parse_units("g_COD/m3")}) == 1
    assert "COD" in str(parse_units("g_COD/m3"))
    assert str(DIMENSIONLESS) == "-"


# --- the unit-string parser -------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("g_COD/m3", {"g": 1, "COD": 1, "m": -3}),
    ("g_N/m3", {"g": 1, "N": 1, "m": -3}),
    ("g_O2/m3", {"g": 1, "O2": 1, "m": -3}),
    ("1/d", {"d": -1}),
    ("mol/L", {"mol": 1, "L": -1}),
    ("M-1 s-1", {"mol": -1, "L": 1, "s": -1}),          # M normalised to mol/L
    ("m3/(g_COD*d)", {"m": 3, "g": -1, "COD": -1, "d": -1}),
    ("gCOD/gCOD", {}),                                   # dimensionless ratio
    ("-", {}),
    ("m³", {"m": 3}),                                    # display superscripts
    ("M⁻¹ s⁻¹", {"mol": -1, "L": 1, "s": -1}),
    ("bar*m3/kmol", {"bar": 1, "m": 3, "kmol": -1}),     # ADM1 gas-law constant
    ("kmol/m3/bar", {"kmol": 1, "m": -3, "bar": -1}),    # Henry constant
])
def test_parse_known_dialects(text, expected):
    dim = parse_units(text)
    assert dim is not None
    assert dim.as_dict() == {k: Fraction(v) for k, v in expected.items()}


def test_parse_half_order_exponent():
    # caret exponents (the WATS half-order biofilm kinetics) parse to fractions
    dim = parse_units("gO2^0.5/m/d")
    assert dim is not None
    assert dim.as_dict()["O2"] == Fraction(1, 2)


@pytest.mark.parametrize("text", [
    "",                       # blank -> undeclared / unknown
    "g COD.m-3",              # the dotted ADM dialect: skipped, not flagged
    "kmol HCO3-.m-3",
    "g//d",                   # malformed
    "???",
])
def test_parse_unknown_returns_none(text):
    assert parse_units(text) is None


def test_distinct_currencies_are_unequal():
    # the whole point: g_COD/m3 and g_N/m3 are NOT the same dimension
    assert parse_units("g_COD/m3") != parse_units("g_N/m3")


# --- the check rules (hand-built dimension maps) ----------------------------

_SD = {
    "X": parse_units("g_COD/m3"),
    "Y": parse_units("g_N/m3"),
    "Z": parse_units("g_COD/m3"),
}
_PD = {
    "R.k": parse_units("1/d"),
    "R.K_cod": parse_units("g_COD/m3"),
    "R.K_n": parse_units("g_N/m3"),
}


def _check(expr, **kw):
    return check_rate_units(parse_rate_expression(expr), "R", _SD, _PD, {}, **kw)


def test_clean_rate_passes():
    # k[1/d] * monod(X, K_cod)[-] * X[g_COD/m3] -> g_COD/m3/d : canonical
    assert _check("k * monod([X], K_cod) * [X]") == []


def test_cross_currency_sum_flagged():
    ws = _check("k * ([X] + [Y]) * [X]", check_root=False)
    assert any("operands" in w.location for w in ws)
    assert "COD" in str(ws) and "N" in str(ws)


def test_monod_currency_mismatch_flagged():
    # saturation argument X is COD, half-saturation K_n is N -> mismatch
    ws = _check("k * monod([X], K_n) * [X]", check_root=False)
    assert any("Monod" in w.location for w in ws)


def test_dropped_concentration_factor_flagged_by_root():
    # forgot the [X] factor: k[1/d] * monod[-] -> 1/d, not currency/volume/time
    ws = _check("k * monod([X], K_cod)")
    assert any(w.location == "rate root" for w in ws)


def test_wrong_root_currency_power_flagged():
    # k[1/d] * X * X -> g_COD^2/m6/d : not a concentration rate
    ws = _check("k * [X] * [X]")
    assert any(w.location == "rate root" for w in ws)


def test_max_shared_units_passes():
    # max of two COD concentrations is a COD concentration; * k[1/d] -> valid rate
    assert _check("k * max([X], [Z])") == []


def test_max_literal_operand_adopts_sibling():
    # the bare 0 in a ``max(0, .)`` clip adopts the sibling's dimension silently
    assert _check("k * max(0, [X])") == []


def test_max_currency_mismatch_flagged():
    # max() operands in different currencies (COD vs N) -> dimensional warning
    ws = _check("k * max([X], [Y])", check_root=False)
    assert any("max" in w.location for w in ws)
    assert "COD" in str(ws) and "N" in str(ws)


def test_safe_div_divides_units():
    # safe_div([X],[Z]) (COD/COD) is dimensionless; * k * [X] -> valid rate
    assert _check("k * safe_div([X], [Z]) * [X]") == []
    # without the [X] factor it is only 1/d -> flagged at the root
    ws = _check("k * safe_div([X], [Z])")
    assert any(w.location == "rate root" for w in ws)


def test_regularizer_constant_not_flagged():
    # a bare numeric guard added to a concentration is intentional, not a bug
    assert _check("k * [X] / ([X] + [Z] + 1.0e-6)", check_root=False) == []


def test_unknown_units_are_skipped():
    # an undeclared (None) species unit propagates as unknown: no false alarm
    sd = {"X": None}
    ws = check_rate_units(parse_rate_expression("[X] + k"), "R",
                          sd, {"R.k": None}, {})
    assert ws == []


def test_check_root_toggle():
    # the dropped-factor case is a root issue only; turning off the root check
    # silences it while the local rules still run
    expr = "k * monod([X], K_cod)"
    assert any(w.location == "rate root" for w in _check(expr))
    assert _check(expr, check_root=False) == []


# --- the full load -> check_units path --------------------------------------

def test_units_metadata_carried_through():
    net = aquakin.load_network("asm1")
    # the new ConditionSpec.units field and the parameter units reach compile
    assert isinstance(net.condition_units, dict)
    assert isinstance(net.parameter_units, dict)
    assert net.parameter_units  # ASM1 declares parameter units


@pytest.mark.parametrize("name", [
    "asm1", "asm2d", "asm3", "asm3_biop", "ozone_bromate", "uv_h2o2",
    "wats_sewer",
])
def test_shipped_networks_are_unit_clean(name):
    # the correctly-annotated shipped networks raise no warning (the check is
    # advisory, so this is the no-false-positive guard, not a proof of math).
    # The SUMO-derived asm2d/asm3/asm3_biop carry real units (issue #199), so
    # they are now actually checked, not skipped, and come out clean.
    assert aquakin.load_network(name).check_units() == []


def test_sumo_networks_have_real_units():
    # The four SUMO-derived ASM networks were imported with placeholder parameter
    # units ("0"/"SmallNumber"/"-BigNumber") and an unparseable species-unit
    # dialect; the shipped YAMLs carry real units, so their rate constants declare
    # a time unit (the inverse-time token parses) and the species units parse.
    from aquakin.utils.units import parse_units
    for name in ("asm2d", "asm2d_tud", "asm3", "asm3_biop"):
        net = aquakin.load_network(name)
        assert "0" not in net.parameter_units.values()
        # at least one rate constant declares 1/d
        assert any(parse_units(u) == parse_units("1/d")
                   for u in net.parameter_units.values())
        # every species unit parses (no leftover dotted dialect)
        assert all(parse_units(net.units_of(s)) is not None for s in net.species)


def test_asm2d_tud_warnings_confined_to_pp_storage():
    # ASM2d-TUD is unit-clean except its two biomass-normalised PP-storage rates
    # (qPP * XPAO**2 / XPP * ...), whose root is irreducibly cross-currency
    # (COD^2/P) -- a property of that model's rate form, which the root check
    # correctly surfaces. Guards that nothing else regresses.
    ws = aquakin.load_network("asm2d_tud").check_units()
    storage = {"Anoxic_storage_of_XPP", "Aerobic_storage_of_XPP"}
    assert ws, "expected the PP-storage root finding to be surfaced"
    assert {w.reaction for w in ws} <= storage


def test_adm1_warnings_confined_to_gas_headspace():
    # ADM1's dissolved/biological reactions are clean; the only warnings are on
    # the gas-headspace pressure sum, which mixes COD-carried H2/CH4 with
    # carbon-carried CO2 (a documented BSM2 gas-phase unit characteristic, not a
    # model error). Guards that the check does not leak into the biology.
    ws = aquakin.load_network("adm1").check_units()
    gas = {"gas_outflow_h2", "gas_outflow_ch4", "gas_outflow_co2"}
    assert ws, "expected the gas-headspace finding to be surfaced"
    assert {w.reaction for w in ws} <= gas


def test_check_network_units_free_function_matches_method():
    # The exported free function aquakin.check_network_units(net) is the
    # implementation the CompiledNetwork.check_units() method delegates to, so
    # the two must return the same findings -- including under check_root=False.
    net = aquakin.load_network("adm1")          # has nonzero (gas-headspace) findings
    via_fn = aquakin.check_network_units(net)
    via_method = net.check_units()
    assert via_fn == via_method
    assert via_fn, "adm1 should surface the gas-headspace findings"
    # the check_root toggle threads through identically
    assert (aquakin.check_network_units(net, check_root=False)
            == net.check_units(check_root=False))
    # a clean network gives an empty list through the free function too
    assert aquakin.check_network_units(aquakin.load_network("asm1")) == []


def _broken_yaml(rate):
    return f"""
network:
  name: broken
  version: "1.0"
  description: "deliberately dimensionally inconsistent rate"
species:
  - name: A
    units: g_COD/m3
    default_concentration: 1.0
  - name: B
    units: g_N/m3
    default_concentration: 1.0
conditions: []
reactions:
  - name: R
    rate: "{rate}"
    parameters:
      k:
        value: 0.1
        units: "1/d"
    stoichiometry:
      A: -1
      B: +1
"""


def test_broken_network_dropped_factor(tmp_path):
    # "k * monod(A, A)" forgets the concentration factor -> 1/d root
    p = tmp_path / "broken.yaml"
    p.write_text(_broken_yaml("k * monod([A], [A])"))
    net = aquakin.load_network_from_file(str(p))
    assert any(w.location == "rate root" for w in net.check_units())


def test_broken_network_cross_currency_sum(tmp_path):
    # A is COD, B is N: adding them is the currency bug an SI check misses
    p = tmp_path / "broken2.yaml"
    p.write_text(_broken_yaml("k * ([A] + [B])"))
    net = aquakin.load_network_from_file(str(p))
    assert any("operands" in w.location for w in net.check_units())


_MIXED_TIME_YAML = """
network:
  name: mixed_time
  description: "Two rate constants in different time units."
species:
  - {name: A, units: g_COD/m3, default_concentration: 1.0}
  - {name: B, units: g_COD/m3, default_concentration: 0.0}
conditions: []
reactions:
  - name: R1
    rate: "k1 * [A]"
    parameters: {k1: {value: 0.1, units: "1/d"}}
    stoichiometry: {A: -1, B: +1}
  - name: R2
    rate: "k2 * [B]"
    parameters: {k2: {value: 0.1, units: "1/s"}}
    stoichiometry: {B: -1, A: +1}
"""


def test_mixed_time_units_flagged(tmp_path):
    """Rate constants in different time units make the RHS dimensionally
    inconsistent (terms summed on different time bases); each rate passes its own
    root check, so the disagreement is flagged once at network scope."""
    p = tmp_path / "mixed_time.yaml"
    p.write_text(_MIXED_TIME_YAML)
    net = aquakin.load_network_from_file(str(p))
    ws = net.check_units()
    network_ws = [w for w in ws if w.reaction == "(network)"
                  and w.location == "time unit"]
    assert len(network_ws) == 1
    assert "1/d" in network_ws[0].detail and "1/s" in network_ws[0].detail


def test_consistent_time_units_not_flagged(tmp_path):
    """A network whose rate constants share one time unit gets no network-scope
    time-unit warning (the shipped networks are already covered above)."""
    p = tmp_path / "ok_time.yaml"
    p.write_text(_MIXED_TIME_YAML.replace('units: "1/s"', 'units: "1/d"'))
    net = aquakin.load_network_from_file(str(p))
    assert not [w for w in net.check_units() if w.location == "time unit"]
