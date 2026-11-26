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
