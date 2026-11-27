"""Every shipped builder gives its layered settler an explicit TSS weight for
each particulate it settles.

``TakacsClarifier`` falls back to a weight of 1.0 for a particulate missing
from its ``tss_factors``. For an ASM1 plant the defaults name every settled
species, but a model with other particulates (ASM2d's XPAO, XPP in g P, the
lumped XTSS itself) would then add COD, phosphorus and TSS into the TSS that
sets the settling velocity, silently. This test fails on such a builder.
"""

import pytest

import aquakin
from aquakin.plant import Aeration
from aquakin.plant.a2o import build_a2o
from aquakin.plant.bsm import build_bsm1, build_bsm2
from aquakin.plant.takacs import TakacsClarifier


def _bsm1_with(model_name):
    """BSM1 on a model whose oxygen is ``SO2``, aerated as its docstring shows."""
    aer = Aeration(do_setpoint=2.0, species="SO2")
    return build_bsm1(
        model=aquakin.load_model(model_name), aeration={f"tank{i}": aer for i in (3, 4, 5)}
    )


BUILDERS = {
    "bsm1": lambda: build_bsm1(),
    "bsm1_asm3": lambda: _bsm1_with("asm3"),
    "bsm1_asm3_2step_n2o": lambda: _bsm1_with("asm3_2step_n2o"),
    "bsm2": lambda: build_bsm2(),
    "a2o": lambda: build_a2o(),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_shipped_settler_weights_every_particulate_explicitly(name):
    plant = BUILDERS[name]()
    settlers = [u for u in plant.units.values() if isinstance(u, TakacsClarifier)]
    assert settlers, f"{name} builds no layered settler"
    for settler in settlers:
        missing = sorted(set(settler.particulate_species) - set(settler.tss_factors))
        assert not missing, (
            f"{name}: settler {settler.name!r} has no TSS factor for {missing}, "
            "so they fall back to 1.0"
        )


def test_the_check_notices_a_settler_without_factors():
    """A bare ASM2d settler with the default (ASM1) factors is what this test
    exists to catch: its particulates are missing from ``tss_factors``."""
    model = aquakin.load_model("asm2d")
    parts = ["XI", "XS", "XH", "XPAO", "XPP", "XPHA", "XAUT", "XTSS"]
    settler = TakacsClarifier(
        name="s", model=model, area=1500.0, height=4.0, underflow_Q=1.8e4,
        particulate_species=parts,
    )
    missing = set(settler.particulate_species) - set(settler.tss_factors)
    assert missing == {"XH", "XPAO", "XPP", "XPHA", "XAUT", "XTSS"}


def test_iron_precipitates_weigh_their_molar_mass_in_the_settler():
    """asm2d_chemp's iron solids are in mol/m3; their TSS factors (from the
    stoichiometry) are molar masses giving g TSS, the TSS state's unit, so the
    settler applies them as they are and the unit audit has nothing to report."""
    from aquakin.plant._constants import lumped_tss_weights

    model = aquakin.load_model("asm2d_chemp")
    parts = [s for s in model.species if s.startswith("X")]
    settler = TakacsClarifier(
        name="s", model=model, area=1500.0, height=4.0, underflow_Q=1.8e4,
        particulate_species=parts, tss_factors=lumped_tss_weights(model),
    )
    factors = dict(zip(parts, settler._part_tss_factors, strict=True))
    assert factors["X_FePO4"] == pytest.approx(150.82, rel=1e-12)
    assert factors["X_FeOH3"] == pytest.approx(106.87, rel=1e-12)
    assert settler.assumed_units() == {}


_MOLAR_TSS = """
model: {name: molar_tss, version: "1.0", description: "test", references: []}
species:
  - {name: S,    units: "g_COD/m3", default_concentration: 1.0}
  - {name: XB,   units: "g_COD/m3", default_concentration: 1.0}
  - {name: XTSS, units: "mol/m3",   default_concentration: 1.0}
conditions: [{name: T, default: 293.15}]
reactions:
  - name: growth
    rate: "k * [S]"
    parameters: {k: {value: 1.0}}
    stoichiometry: {S: -1, XB: 1, XTSS: 0.01}
"""


def test_a_tss_state_in_moles_is_reported_by_the_unit_audit(tmp_path):
    """The settling parameters are per g/m3; a TSS state in mol/m3 cannot be
    put on that basis, so the settler reports it as an assumed unit."""
    path = tmp_path / "molar_tss.yaml"
    path.write_text(_MOLAR_TSS)
    model = aquakin.load_model_from_file(path)
    settler = TakacsClarifier(
        name="s", model=model, area=100.0, height=4.0, underflow_Q=100.0,
        particulate_species=["XB", "XTSS"], tss_factors={"XB": 0.01, "XTSS": 0.0},
    )
    assert settler.assumed_units() == {"XTSS": "g/m3"}
