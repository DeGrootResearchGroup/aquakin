"""The plant-wide nitrite level (``plantwide_nitrite``) on BSM1.

The level runs on ``build_bsm1`` (open loop, Takács settler) with BSM1's
constant influent carried into its species (``examples/plantwide/bsm1``; ASM1's
nitrifiers become AOB and NOB in the ratio of their yields, phosphate is given
in excess). Its nitrifiers have the kinetics of Hiatt & Grady (2008).
Checked:

- from its steady state, the plant conserves nitrogen, phosphorus and COD over
  10 days: what the influent brings equals what leaves in the effluent and
  wastage, plus the gases the aeration strips (N2, and the oxygen taken up);
- it nitrifies through nitrite: effluent ammonium below 2 g N/m3, the AOB
  outnumbering the NOB, and both nitrite and nitrate in the effluent.

For reference, the core level's one-step nitrifiers on the same plant leave
1.11 g N/m3 of ammonium and 11.0 of nitrate; this level 1.77 of ammonium, 5.4
of nitrite and 2.3 of nitrate. Nitrite builds up because BSM1's last tank
runs at 0.5 g O2/m3 and the NOB's oxygen half-saturation (1.2) is twice the
AOB's: the NOB only just outgrow their washout. With the NOB's half-saturation
at the AOB's 0.6 the effluent holds 1.1 of nitrite and 9.1 of nitrate.

About 2 minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

import aquakin

pytestmark = pytest.mark.validation

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "plantwide" / "bsm1"))


@pytest.fixture(scope="module")
def nitrite_bsm1():
    import bsm1_comparison as b

    model = aquakin.load_model_from_file(Path(b.CORE).parent / "plantwide_nitrite.yaml")
    result = b.run(model)
    solution = result["plant"].solve(
        t_span=(0.0, 10.0),
        t_eval=jnp.linspace(0.0, 10.0, 11),
        y0=result["steady"].state,
        rtol=1e-8,
        atol=1e-6,
    )
    return b, model, result, solution


def test_plant_conserves_nitrogen_phosphorus_and_cod(nitrite_bsm1):
    """Each balance to 1e-9 of the inflow: the plant is at steady state, the
    fluxes are integrated over 11 saved times, and the solve's rtol is 1e-8."""
    _, _, result, solution = nitrite_bsm1
    assert bool(result["steady"].converged)
    balance = result["plant"].mass_balance(solution, components=("N", "P", "COD"))
    for q in ("N", "P", "COD"):
        assert abs(balance[q].imbalance) < 1e-9 * balance[q].inflow, q
    assert balance["N"].gas > 0.25 * balance["N"].inflow  # denitrified N leaves as N2


def test_level_nitrifies_through_nitrite_on_bsm1(nitrite_bsm1):
    b, model, result, _ = nitrite_bsm1
    tank5 = result["plant"].states_by_unit(result["steady"].state)["tank5"]

    def at(species):
        return float(tank5[model.species_index[species]])

    assert at("SNH4") < 2.0
    assert at("SNO2") > 1.0
    assert at("SNO3") > 1.0
    assert at("XAOB") > at("XNOB") > 1.0
