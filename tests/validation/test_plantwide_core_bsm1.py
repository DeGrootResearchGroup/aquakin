"""The plant-wide core (``plantwide_core``) on BSM1, against ASM2d.

Both models run on ``build_bsm1`` (open loop, ideal clarifier) with BSM1's
constant dry-weather influent carried into their species
(``examples/plantwide/bsm1``). Checked:

- with the processes the two models do not share switched off (the core's
  fermentation and slower anaerobic decay; ASM2d's fermentation, with its
  anaerobic hydrolysis factor set to the core's) and without ASM2d's
  positivity limiter, the two plants' steady states are the same: the core is
  ASM2d at plant scale wherever their processes coincide. ASM2d's limiter
  (threshold 1e-3) alone moves its steady state by 0.2%: in the anoxic tanks
  oxygen sits below the threshold, so the limiter throttles its uptake;
- the core as shipped nitrifies and denitrifies on BSM1, and its heterotrophs
  oxidize the propionate and hydrogen its fermentation makes in the anoxic
  tanks (the hydrogen is stripped by the air, as the core declares it a gas);
  its effluent nitrate stays within 1 g N/m3 of ASM2d's.

About 2 minutes (four steady states).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

import aquakin

pytestmark = pytest.mark.validation

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "plantwide" / "bsm1"))


@pytest.fixture(scope="module")
def comparison():
    import bsm1_comparison as b

    core = aquakin.load_model_from_file(b.CORE)
    asm2d = dataclasses.replace(aquakin.load_model("asm2d"), positivity_threshold=None)
    etafe = float(core.default_parameters()[core.param_index["etafe"]])

    def aligned(model):
        plant, _ = b.bsm1_plant(model)
        return b.run(model, b.aligned_parameters(plant, model, etafe))

    return dict(
        module=b,
        core_aligned=aligned(core),
        asm2d_aligned=aligned(asm2d),
        core=b.run(core),
        asm2d=b.run(aquakin.load_model("asm2d")),
    )


def test_core_equals_asm2d_on_bsm1_where_their_processes_coincide(comparison):
    """Tank 5 (whose solubles are the effluent's) to rtol 1e-5: each steady
    state is converged to a scaled residual of 1e-9."""
    b = comparison["module"]
    for result in (comparison["core_aligned"], comparison["asm2d_aligned"]):
        assert bool(result["steady"].converged)
    core, asm2d = b.summary(comparison["core_aligned"]), b.summary(comparison["asm2d_aligned"])
    for s in ("SNH4", "SNO3", "SF", "SO2", "XH", "XAUT", "XS", "XI"):
        assert core[s] == pytest.approx(asm2d[s], rel=1e-5), s


def test_core_as_shipped_treats_bsm1_and_oxidizes_its_fermentation_products(comparison):
    """Effluent ammonium below 2 g N/m3 (nitrification); propionate and
    hydrogen below 0.05 g COD/m3 (without heterotroph growth on propionate,
    1.3 g/m3 of it left the plant). The core's effluent nitrate is higher than
    ASM2d's (11.0 against 10.4 g N/m3), mostly because its fermentation in the
    anoxic tanks turns some substrate into hydrogen, which is stripped rather
    than used for denitrification; it is compared to 1 g N/m3."""
    b = comparison["module"]
    assert bool(comparison["core"]["steady"].converged)
    core, asm2d = b.summary(comparison["core"]), b.summary(comparison["asm2d"])
    assert core["SNH4"] < 2.0
    assert core["SPRO"] < 0.05
    assert core["SH2"] < 0.05
    assert core["SNO3"] == pytest.approx(asm2d["SNO3"], abs=1.0)
