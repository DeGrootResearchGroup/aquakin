"""BSM1 with a model other than ASM1: clarifier solids and per-tank aeration.

``solids_species`` picks the settling and TSS species for the BSM1 clarifier:
the ASM1 constants when the model's particulates are exactly ASM1's, otherwise
every ``X``-prefixed species settles and those carrying COD are TSS (read from
the declared units when the model has no composition table). ``build_bsm1`` takes
explicit ``settling_species`` / ``tss_species`` and per-tank ``aeration``
overrides. ``lumped_tss_weights`` reads the TSS weights of a model that tracks
its TSS as a state from its stoichiometry. These tests only build plants;
nothing is solved.
"""

from pathlib import Path

import pytest

import aquakin
from aquakin.plant import Aeration
from aquakin.plant._constants import (
    ASM1_SETTLING_SPECIES,
    ASM1_TSS_FACTOR,
    ASM1_TSS_SPECIES,
    lumped_tss_weights,
    solids_species,
    with_lumped_tss,
)
from aquakin.plant.bsm import build_bsm1
from tests.conftest import TSS_FACTOR_PARAMS

FIXTURES = Path(__file__).parent.parent / "fixtures"

# A two-particulate model with declared composition: XB carries COD, XN only N.
_DECLARED = """
model: {name: solids_declared, version: "1.0", description: "test", references: []}
species:
  - {name: SS, units: "g_COD/m3", default_concentration: 1.0, composition: {COD: 1.0}}
  - {name: XB, units: "g_COD/m3", default_concentration: 1.0, composition: {COD: 1.0, N: 0.08}}
  - {name: XN, units: "g_N/m3",   default_concentration: 1.0, composition: {N: 1.0}}
conditions: [{name: T, default: 293.15}]
reactions:
  - name: growth
    rate: "k * [SS]"
    parameters: {k: {value: 1.0}}
    stoichiometry: {SS: -1, XB: 1}
"""

# The same shape with no composition anywhere (and no shipped table).
_BARE = """
model: {name: solids_bare, version: "1.0", description: "test", references: []}
species:
  - {name: SS,  units: "g_COD/m3", default_concentration: 1.0}
  - {name: XB,  units: "g_COD/m3", default_concentration: 1.0}
  - {name: XND, units: "g_N/m3",   default_concentration: 1.0}
conditions: [{name: T, default: 293.15}]
reactions:
  - name: growth
    rate: "k * [SS]"
    parameters: {k: {value: 1.0}}
    stoichiometry: {SS: -1, XB: 1}
"""

# ASM1 plus one extra particulate: still ASM1-named, but not ASM1's solids.
_ASM1_PLUS = """
model:
  name: asm1_plus_particulate
  version: "1.0"
  description: "test"
  references: []
  extends: asm1
species:
  - {name: XEXTRA, units: "g_COD/m3", default_concentration: 0.0}
"""


def _load(tmp_path, name, text):
    path = tmp_path / f"{name}.yaml"
    path.write_text(text)
    return aquakin.load_model_from_file(path)


# --- solids_species --------------------------------------------------------------


def test_asm1_keeps_its_constants():
    settling, tss = solids_species(aquakin.load_model("asm1"))
    assert settling == ASM1_SETTLING_SPECIES
    assert tss == ASM1_TSS_SPECIES


def test_tss_from_declared_composition(tmp_path):
    settling, tss = solids_species(_load(tmp_path, "declared", _DECLARED))
    assert settling == ("XB", "XN")
    assert tss == ("XB",)  # XN carries no COD


def test_tss_without_composition_excludes_xnd(tmp_path):
    settling, tss = solids_species(_load(tmp_path, "bare", _BARE))
    assert settling == ("XB", "XND")
    assert tss == ("XB",)


# No composition, particulates declared in several currencies (#672).
_MIXED_UNITS = """
model: {name: solids_mixed, version: "1.0", description: "test", references: []}
species:
  - {name: SS,    units: "g_COD/m3", default_concentration: 1.0}
  - {name: XB,    units: "g_COD/m3", default_concentration: 1.0}
  - {name: XKG,   units: "kgCOD/m3", default_concentration: 1.0}
  - {name: XPP,   units: "g_P/m3",   default_concentration: 1.0}
  - {name: XTSS,  units: "g_TSS/m3", default_concentration: 1.0}
  - {name: XMEP,  units: "mol/m3",   default_concentration: 1.0}
  - {name: XODD,  units: "furlongs", default_concentration: 1.0}
  - {name: XND,   units: "furlongs", default_concentration: 1.0}
conditions: [{name: T, default: 293.15}]
reactions:
  - name: growth
    rate: "k * [SS]"
    parameters: {k: {value: 1.0}}
    stoichiometry: {SS: -1, XB: 1}
"""


def test_tss_without_composition_reads_declared_units(tmp_path):
    # Only COD-mass particulates count; precipitates in moles, stored P and a
    # lumped TSS state do not. An unreadable unit falls back to the name rule.
    settling, tss = solids_species(_load(tmp_path, "mixed", _MIXED_UNITS))
    assert settling == ("XB", "XKG", "XPP", "XTSS", "XMEP", "XODD", "XND")
    assert tss == ("XB", "XKG", "XODD")


def test_asm2d_chemp_tss_matches_asm2d():
    # asm2d_chemp has no composition table; its TSS species now match those
    # asm2d gets from its table, and the Takacs settler audits clean.
    chemp = aquakin.load_model("asm2d_chemp")
    assert solids_species(chemp)[1] == solids_species(aquakin.load_model("asm2d"))[1]
    aeration = {f"tank{i}": Aeration(kla=240.0, species="SO2") for i in (3, 4, 5)}
    plant = build_bsm1(chemp, use_takacs=True, aeration=aeration)
    assert plant.check_units(include_models=False).ok


def test_extra_particulate_on_asm1_settles(tmp_path):
    # A model holding every ASM1 particulate plus another must not fall back to
    # the ASM1 constants, which would leave the extra solid in the effluent.
    model = _load(tmp_path, "plus", _ASM1_PLUS)
    settling, tss = solids_species(model)
    assert "XEXTRA" in settling
    assert set(ASM1_SETTLING_SPECIES) <= set(settling)
    assert "XND" not in tss
    assert set(ASM1_TSS_SPECIES) <= set(tss)


def test_model_without_particulates_raises():
    toy = aquakin.load_model_from_file(FIXTURES / "stripped_gas_model.yaml")
    with pytest.raises(ValueError, match="no particulate"):
        solids_species(toy)


# --- build_bsm1 overrides --------------------------------------------------------


def test_build_bsm1_per_tank_aeration_override():
    plant = build_bsm1(aeration={"tank1": Aeration(kla=10.0), "tank4": None})
    assert plant.units["tank1"].aeration.kla == 10.0
    assert plant.units["tank2"].aeration is None  # anoxic by default
    assert plant.units["tank4"].aeration is None  # aerated by default, switched off
    assert plant.units["tank5"].aeration.kla == pytest.approx(84.0)  # reference kLa


def test_build_bsm1_unknown_aeration_tank_raises():
    with pytest.raises(ValueError, match="not BSM1 tanks"):
        build_bsm1(aeration={"tank6": Aeration(kla=10.0)})


@pytest.mark.parametrize("use_takacs", [False, True])
def test_build_bsm1_explicit_solids(use_takacs):
    settling = ("XS", "XI", "XB_H", "XB_A", "XP", "XND")
    tss = ("XS", "XB_H")
    plant = build_bsm1(use_takacs=use_takacs, settling_species=settling, tss_species=tss)
    clar = plant.units["clarifier"]
    assert list(clar.particulate_species) == list(settling)
    if use_takacs:
        assert clar.tss_factors == {sp: (ASM1_TSS_FACTOR if sp in tss else 0.0) for sp in settling}


@pytest.mark.parametrize("name", ["asm2d", "asm3"])
def test_build_bsm1_weighs_a_lumped_tss_models_solids_by_its_own_factors(name):
    """On a model that tracks its TSS as a state, the BSM1 settler weighs each
    particulate by the model's own iTSS / iSS parameter (the TSS state at 0, a
    g TSS precipitate at 1), as build_a2o does, not ASM1's 0.75 per g COD with
    stored phosphate at 0. Given ``tss_species``, the 0.75 weighting stands."""
    model = aquakin.load_model(name)
    tss_state, factors = TSS_FACTOR_PARAMS[name]
    p = dict(zip(model.parameters, map(float, model.default_parameters()), strict=True))
    aer = {f"tank{i}": Aeration(kla=240.0, species="SO2") for i in (3, 4, 5)}
    clar = build_bsm1(model, aeration=aer).units["clarifier"]
    expected = {sp: p[f] for sp, f in factors.items()} | {tss_state: 0.0}
    expected |= dict.fromkeys(set(clar.particulate_species) - set(expected), 1.0)
    assert set(clar.tss_factors) == set(clar.particulate_species)
    for sp, w in expected.items():
        assert clar.tss_factors[sp] == pytest.approx(w, rel=1e-12, abs=1e-12), sp

    given = ("XI", "XS")
    explicit = build_bsm1(model, aeration=aer, tss_species=given).units["clarifier"]
    assert explicit.tss_factors == {
        sp: (ASM1_TSS_FACTOR if sp in given else 0.0) for sp in clar.particulate_species
    }


def test_build_bsm1_tss_must_settle():
    with pytest.raises(ValueError, match="not settling species"):
        build_bsm1(settling_species=("XS", "XI"), tss_species=("XS", "XB_H"))


# --- lumped_tss_weights ------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TSS_FACTOR_PARAMS))
def test_lumped_tss_weights_are_the_models_tss_factors(name):
    """The weights solved from the stoichiometry are the model's own iTSS / iSS
    parameters, the TSS state weighs 0, and a precipitate declared in g TSS
    (asm2d's XMeOH / XMeP) weighs 1."""
    model = aquakin.load_model(name)
    tss, factors = TSS_FACTOR_PARAMS[name]
    p = dict(zip(model.parameters, map(float, model.default_parameters()), strict=True))
    expected = {s: p[f] for s, f in factors.items()} | {tss: 0.0}
    expected |= {s: 1.0 for s in ("XMeOH", "XMeP") if s in model.species}
    weights = lumped_tss_weights(model)
    assert set(weights) == set(expected)
    for s, w in expected.items():
        assert weights[s] == pytest.approx(w, rel=1e-12, abs=1e-12), s


def test_lumped_tss_weights_follow_the_parameters():
    """A changed iTSS factor in the parameter vector changes its weight."""
    model = aquakin.load_model("asm2d")
    params = model.default_parameters().at[model.param_index["iTSS_XPP"]].set(2.0)
    assert lumped_tss_weights(model, params)["XPP"] == pytest.approx(2.0, rel=1e-12)


def test_with_lumped_tss_sums_the_particulates_and_keeps_a_given_value():
    model = aquakin.load_model("asm2d")
    comp = with_lumped_tss(model, {"XI": 30.0, "XS": 180.0, "XH": 40.0, "SNH4": 25.0})
    assert comp["XTSS"] == pytest.approx(0.75 * 30.0 + 0.75 * 180.0 + 0.9 * 40.0, rel=1e-12)
    assert comp["SNH4"] == 25.0
    assert with_lumped_tss(model, {"XI": 30.0, "XTSS": 7.0})["XTSS"] == 7.0


# Two particulates that are only ever made together: their weights are not separable.
_TOGETHER = """
model: {name: tss_together, version: "1.0", description: "test", references: []}
species:
  - {name: SS,   units: "g_COD/m3", default_concentration: 1.0}
  - {name: XB,   units: "g_COD/m3", default_concentration: 1.0}
  - {name: XC,   units: "g_COD/m3", default_concentration: 1.0}
  - {name: XTSS, units: "g_TSS/m3", default_concentration: 1.0}
conditions: [{name: T, default: 293.15}]
reactions:
  - name: growth
    rate: "k * [SS]"
    parameters: {k: {value: 1.0}}
    stoichiometry: {SS: -2, XB: 1, XC: 1, XTSS: 1.7}
"""

# One particulate whose TSS state changes at two different ratios.
_INCONSISTENT = """
model: {name: tss_inconsistent, version: "1.0", description: "test", references: []}
species:
  - {name: SS,   units: "g_COD/m3", default_concentration: 1.0}
  - {name: XB,   units: "g_COD/m3", default_concentration: 1.0}
  - {name: XTSS, units: "g_TSS/m3", default_concentration: 1.0}
conditions: [{name: T, default: 293.15}]
reactions:
  - name: growth
    rate: "k * [SS]"
    parameters: {k: {value: 1.0}}
    stoichiometry: {SS: -1, XB: 1, XTSS: 0.9}
  - name: decay
    rate: "b * [XB]"
    parameters: {b: {value: 0.1}}
    stoichiometry: {SS: 1, XB: -1, XTSS: -0.5}
"""


@pytest.mark.parametrize(
    ("name", "text", "match"),
    [
        ("bare", _BARE, "has no TSS state"),
        ("together", _TOGETHER, "does not determine the TSS weight"),
        ("inconsistent", _INCONSISTENT, "no particulate TSS weights reproduce"),
    ],
)
def test_lumped_tss_weights_refuses_what_the_model_does_not_determine(
    tmp_path, name, text, match
):
    """Each refusal says to give the settler's TSS factors explicitly; doing so
    builds a settler that weighs the solids as given."""
    from aquakin.plant import TakacsClarifier

    model = _load(tmp_path, name, text)
    with pytest.raises(ValueError, match=match):
        lumped_tss_weights(model)
    parts = [s for s in model.species if s.startswith("X")]
    clar = TakacsClarifier(
        "c",
        model,
        area=1.0,
        height=4.0,
        underflow_Q=1.0,
        particulate_species=parts,
        tss_factors=dict.fromkeys(parts, 0.8),
    )
    assert clar._part_tss_factors == [0.8] * len(parts)
