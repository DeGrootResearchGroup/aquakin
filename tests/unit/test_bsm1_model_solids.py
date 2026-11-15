"""BSM1 with a model other than ASM1: clarifier solids and per-tank aeration.

``solids_species`` picks the settling and TSS species for the BSM1 clarifier:
the ASM1 constants when the model's particulates are exactly ASM1's, otherwise
every ``X``-prefixed species settles and those carrying COD are TSS (every one
but ``XND`` when the model has no composition table). ``build_bsm1`` takes
explicit ``settling_species`` / ``tss_species`` and per-tank ``aeration``
overrides. These tests only build plants; nothing is solved.
"""

from pathlib import Path

import pytest

import aquakin
from aquakin.plant import Aeration
from aquakin.plant._constants import (
    ASM1_SETTLING_SPECIES,
    ASM1_TSS_FACTOR,
    ASM1_TSS_SPECIES,
    solids_species,
)
from aquakin.plant.bsm import build_bsm1

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


def test_build_bsm1_tss_must_settle():
    with pytest.raises(ValueError, match="not settling species"):
        build_bsm1(settling_species=("XS", "XI"), tss_species=("XS", "XB_H"))
