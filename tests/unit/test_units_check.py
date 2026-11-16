"""Dimensional ('unit') consistency checking for rate expressions (issue #161).

Three layers: the currency-token :class:`Dimension` algebra, the tolerant
unit-string parser, and the propagation/check over a rate AST. The check must
(a) stay silent on the correctly-annotated shipped models (no false alarm) and
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
    ("mg/L", {"mg": 1, "L": -1}),                        # milli-prefixes are symbols,
    ("mmol/L", {"mmol": 1, "L": -1}),                    # not metre x gram / mole
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


@pytest.mark.parametrize("expr", [
    "k * (0.1 / (0.1 + [X])) * [Z]",
    "k * [Z] * (0.1 / ([X] + 0.1))",
    "k * (0.1 / (0.1 + [X] + [Z])) * [Z]",
])
def test_literal_half_saturation_is_dimensionless(expr):
    # c / (c + X) is an inhibition switch whose half-saturation constant is an
    # inline literal: the literal carries X's units, so the term is
    # dimensionless and the rate stays a concentration per time. (The pattern
    # is matched as its own subexpression -- a named expression or parentheses
    # -- since ``k * c / (c + X)`` parses as ``(k * c) / (c + X)``.)
    assert _check(expr) == []


def test_literal_numerator_without_matching_term_still_divides():
    # A different literal in the numerator is not the K / (K + X) pattern, so
    # the ratio keeps its 1 / [X] units and the root check still sees them.
    assert any(w.location == "rate root" for w in _check("k * (0.2 / (0.1 + [X])) * [Z]"))


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
    net = aquakin.load_model("asm1")
    # the new ConditionSpec.units field and the parameter units reach compile
    assert isinstance(net.condition_units, dict)
    assert isinstance(net.parameter_units, dict)
    assert net.parameter_units  # ASM1 declares parameter units


def _shipped_model_names():
    from importlib.resources import files

    return sorted(
        f.name[: -len(".yaml")]
        for f in files("aquakin.models").iterdir()
        if f.name.endswith(".yaml")
    )


@pytest.mark.parametrize("name", _shipped_model_names())
def test_shipped_models_are_unit_clean(name):
    # Every shipped model is unit-clean: no finding outside the reactions it
    # explicitly acknowledges (``units_acknowledged:``, each with a written
    # reason). A new or edited model that trips the check must either fix its
    # units or acknowledge the reaction deliberately. The check is advisory, so
    # this is the no-false-positive guard, not a proof of the math.
    net = aquakin.load_model(name)
    assert net.check_units() == []
    # Every acknowledgement carries a reason and names a real reaction.
    for reaction, reason in net.units_acknowledged.items():
        assert reaction in net.reaction_names
        assert reason.strip()
    # The plant unit audit relies on each model declaring its time unit.
    assert net.time_unit is not None


def test_acknowledged_reactions_are_pinned():
    # Acknowledging a reaction hides its findings, so the set is pinned here:
    # adding one is a deliberate, reviewed change.
    acknowledged = {
        name: set(aquakin.load_model(name).units_acknowledged)
        for name in _shipped_model_names()
    }
    assert {k: v for k, v in acknowledged.items() if v} == {
        "adm1": {"gas_outflow_h2", "gas_outflow_ch4", "gas_outflow_co2"},
    }


def test_sumo_models_have_real_units():
    # The four SUMO-derived ASM models were imported with placeholder parameter
    # units ("0"/"SmallNumber"/"-BigNumber") and an unparseable species-unit
    # dialect; the shipped YAMLs carry real units, so their rate constants declare
    # a time unit (the inverse-time token parses) and the species units parse.
    from aquakin.utils.units import parse_units
    for name in ("asm2d", "asm2d_tud", "asm3", "asm3_biop"):
        net = aquakin.load_model(name)
        assert "0" not in net.parameter_units.values()
        # at least one rate constant declares 1/d
        assert any(parse_units(u) == parse_units("1/d")
                   for u in net.parameter_units.values())
        # every species unit parses (no leftover dotted dialect)
        assert all(parse_units(net.units_of(s)) is not None for s in net.species)


def test_asm2d_tud_storage_constants_carry_specific_units():
    # The anaerobic-maintenance constant is a specific rate (g P per g COD of
    # PAO per day). The PP-storage rate is inversely proportional to the stored
    # fraction f_PP = XPP/XPAO (g P/g COD), so its constant carries
    # (g P/g COD)^2/d for the rate to be g P/m3/d like its stoichiometry. The
    # charge factors are kmol per gram. With those units every ASM2d-TUD rate
    # checks clean without any acknowledgement.
    from aquakin.utils.units import _Ctx, _infer, parse_units

    net = aquakin.load_model("asm2d_tud")
    assert parse_units(net.parameter_units["mAN"]) == parse_units("g_P/(g_COD*d)")
    assert parse_units(net.parameter_units["qPP"]) == parse_units("g_P*g_P/(g_COD*g_COD*d)")
    dims = [
        {k: parse_units(v) for k, v in getattr(net, attr).items()}
        for attr in ("species_units", "parameter_units", "condition_units")
    ]
    for name, ast in zip(net.reaction_names, net.rate_asts):
        if "storage_of_XPP" in name:
            assert _infer(ast, _Ctx(name, *dims, [])) == parse_units("g_P/m3/d")
    assert parse_units(net.parameter_units["iCharge_SNHx"]) == parse_units("kmol/g_N")
    assert net.units_acknowledged == {}
    assert net.check_units(include_acknowledged=True) == []


def test_adm1_findings_confined_to_gas_headspace():
    # ADM1's dissolved/biological reactions are clean; the only findings are on
    # the gas-headspace pressure sum, which mixes COD-carried H2/CH4 with
    # carbon-carried CO2 (a documented BSM2 gas-phase unit characteristic, not a
    # model error), acknowledged in the YAML. Guards that the check does not leak
    # into the biology.
    net = aquakin.load_model("adm1")
    ws = net.check_units(include_acknowledged=True)
    gas = {"gas_outflow_h2", "gas_outflow_ch4", "gas_outflow_co2"}
    assert ws, "expected the gas-headspace finding to be surfaced"
    assert {w.reaction for w in ws} <= gas
    assert net.check_units() == []


def test_adm1_parameters_all_declare_units():
    from aquakin.utils.units import parse_units

    net = aquakin.load_model("adm1")
    assert all(parse_units(u) is not None for u in net.parameter_units.values())


def test_check_model_units_free_function_matches_method():
    # The exported free function aquakin.check_model_units(net) is the
    # implementation the CompiledModel.check_units() method delegates to, so
    # the two must return the same findings -- including under check_root=False
    # and include_acknowledged=True.
    net = aquakin.load_model("adm1")          # has acknowledged gas findings
    via_fn = aquakin.check_model_units(net, include_acknowledged=True)
    via_method = net.check_units(include_acknowledged=True)
    assert via_fn == via_method
    assert via_fn, "adm1 should surface the gas-headspace findings"
    assert aquakin.check_model_units(net) == net.check_units() == []
    # the check_root toggle threads through identically
    assert (aquakin.check_model_units(net, check_root=False, include_acknowledged=True)
            == net.check_units(check_root=False, include_acknowledged=True))
    # a clean model gives an empty list through the free function too
    assert aquakin.check_model_units(aquakin.load_model("asm1")) == []


def _broken_yaml(rate):
    return f"""
model:
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


def test_broken_model_dropped_factor(tmp_path):
    # "k * monod(A, A)" forgets the concentration factor -> 1/d root
    p = tmp_path / "broken.yaml"
    p.write_text(_broken_yaml("k * monod([A], [A])"))
    net = aquakin.load_model_from_file(str(p))
    assert any(w.location == "rate root" for w in net.check_units())


def test_broken_model_cross_currency_sum(tmp_path):
    # A is COD, B is N: adding them is the currency bug an SI check misses
    p = tmp_path / "broken2.yaml"
    p.write_text(_broken_yaml("k * ([A] + [B])"))
    net = aquakin.load_model_from_file(str(p))
    assert any("operands" in w.location for w in net.check_units())


_MIXED_TIME_YAML = """
model:
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


def test_units_acknowledged_hides_findings_by_default(tmp_path):
    p = tmp_path / "ack.yaml"
    text = _broken_yaml("k * ([A] + [B])").replace(
        '    rate: "k * ([A] + [B])"',
        '    rate: "k * ([A] + [B])"\n    units_acknowledged: "deliberate, for the test"',
    )
    p.write_text(text)
    net = aquakin.load_model_from_file(str(p))
    assert net.units_acknowledged == {"R": "deliberate, for the test"}
    assert net.check_units() == []
    assert any("operands" in w.location for w in net.check_units(include_acknowledged=True))


@pytest.mark.parametrize("a, b, expected", [
    ("mg/L", "g_COD/m3", True),       # same scale; currency unstated on one side
    ("g_COD/m3", "g_COD/m3", True),
    ("kgCOD/m3", "g_COD/m3", False),  # factor 1000
    ("g_N/m3", "g_COD/m3", False),    # both state a currency, and they differ
    ("mmol/L", "mol/m3", True),
    ("m3/h", "m3/d", False),
    ("1/s", "1/d", False),
    ("d", "d", True),
    ("degC", "K", None),              # unparseable -> unknown
    ("", "K", None),
])
def test_units_match(a, b, expected):
    assert aquakin.units_match(a, b) is expected


@pytest.mark.parametrize("a, b, factor", [
    ("kgCOD/m3", "g_COD/m3", 1000.0),
    ("mol/m3", "kmol/m3", 1.0e-3),
    ("mg/L", "g/m3", 1.0),
    ("m3/h", "m3/d", 24.0),
])
def test_conversion_factor(a, b, factor):
    from aquakin.utils import conversion_factor

    assert conversion_factor(a, b) == pytest.approx(factor, rel=1e-12)


def test_conversion_factor_unknown_and_incompatible():
    from aquakin.utils import conversion_factor

    assert conversion_factor("furlongs", "m") is None
    with pytest.raises(ValueError, match="different quantities"):
        conversion_factor("g_N/m3", "kmolN/m3")   # mass vs amount of substance
    with pytest.raises(ValueError, match="different quantities"):
        conversion_factor("g_N/m3", "g_COD/m3")   # two stated currencies


_STOICH_YAML = """
model: {{name: stoich_units, description: "unit-coefficient check"}}
species:
  - {{name: X, units: g_COD/m3, default_concentration: 1.0}}
  - {{name: P, units: g_P/m3, default_concentration: 1.0}}
  - {{name: Q, units: g_P/m3, default_concentration: 1.0}}
conditions: []
parameters:
  k: {{value: 0.1, units: "{k_units}"}}
  y: {{value: 1.0, units: "g_P/g_P"}}
reactions:
  - name: R
    rate: "k * [X]"
    stoichiometry: {{P: -1, Q: {q_coef}}}
"""


def _stoich_model(tmp_path, k_units, q_coef="1"):
    p = tmp_path / "stoich.yaml"
    p.write_text(_STOICH_YAML.format(k_units=k_units, q_coef=q_coef))
    return aquakin.load_model_from_file(str(p))


def test_rate_must_match_a_unit_coefficient_species(tmp_path):
    # P and Q change by exactly the rate (coefficients -1, +1), so the rate
    # must be in g P/m3 per day; with k in 1/d it is in g COD/m3 per day.
    net = _stoich_model(tmp_path, "1/d")
    ws = [w for w in net.check_units() if w.location == "stoichiometry"]
    assert len(ws) == 1 and ws[0].reaction == "R"
    assert "'P'" in ws[0].detail and "'Q'" in ws[0].detail
    assert net.check_units(check_stoichiometry=False) == []
    # The specific-rate unit makes it consistent.
    assert _stoich_model(tmp_path, "g_P/(g_COD*d)").check_units() == []


def test_stoichiometry_check_ignores_expression_coefficients(tmp_path):
    # A coefficient written as an expression is not a fixed +/-1, even when it
    # evaluates to 1, so only P anchors the rate here; P still disagrees.
    net = _stoich_model(tmp_path, "1/d", q_coef='"y"')
    (w,) = [w for w in net.check_units() if w.location == "stoichiometry"]
    assert "'P'" in w.detail and "'Q'" not in w.detail


def test_mixed_time_units_flagged(tmp_path):
    """Rate constants in different time units make the RHS dimensionally
    inconsistent (terms summed on different time bases); each rate passes its own
    root check, so the disagreement is flagged once at model scope."""
    p = tmp_path / "mixed_time.yaml"
    p.write_text(_MIXED_TIME_YAML)
    net = aquakin.load_model_from_file(str(p))
    ws = net.check_units()
    model_ws = [w for w in ws if w.reaction == "(model)"
                  and w.location == "time unit"]
    assert len(model_ws) == 1
    assert "1/d" in model_ws[0].detail and "1/s" in model_ws[0].detail


def test_consistent_time_units_not_flagged(tmp_path):
    """A model whose rate constants share one time unit gets no model-scope
    time-unit warning (the shipped models are already covered above)."""
    p = tmp_path / "ok_time.yaml"
    p.write_text(_MIXED_TIME_YAML.replace('units: "1/s"', 'units: "1/d"'))
    net = aquakin.load_model_from_file(str(p))
    assert not [w for w in net.check_units() if w.location == "time unit"]


def test_parser_unbalanced_parenthesis_raises():
    # parse_units() catches the _ParseError and returns None; drive the
    # underlying product parser directly to exercise the raise itself.
    from aquakin.utils.units import _lex, _ParseError, _ProductParser

    parser = _ProductParser(_lex("(g"))
    with pytest.raises(_ParseError, match="unbalanced parenthesis"):
        parser.parse()


def test_parse_units_accepts_display_dot():
    # The display multiplication dot (U+00B7 and U+22C5) parses as '*', so the
    # prettified form round-trips through the parser.
    target = parse_units("g_COD/m3/d")
    assert target is not None
    assert parse_units("g_COD·m⁻³·d⁻¹") == target
    assert parse_units("g_COD⋅m-3⋅d-1") == target
