"""The plant-wide nitrite level (``plantwide_nitrite``) on BSM1.

The level runs on ``build_bsm1`` (open loop, Takács settler) with BSM1's
constant influent carried into its species (``examples/plantwide/bsm1``; ASM1's
nitrifiers become AOB and NOB in the ratio of their yields, phosphate is given
in excess), at 20 C. Checked:

- from its steady state, the plant conserves nitrogen, phosphorus and COD over
  10 days: what the influent brings equals what leaves in the effluent and
  wastage, plus the gases the aeration strips (N2, and the oxygen taken up);
- it nitrifies through nitrite without accumulating it: effluent ammonium
  below a tenth of the influent's, nitrite below 0.5 g N/m3. Plants at low DO keep their NOB and
  little nitrite (Law et al. 2019: 0.3-0.7 g N/m3 at DO 0.8-1.2 and 4-6 d
  sludge age; Liu et al. 2019: none at DO 0.3), and BSM1's last tank runs at
  about 0.2-0.5 g O2/m3.

This level leaves 1.68 g N/m3 of ammonium, 0.08 of nitrite and 11.76 of
nitrate (the core: 1.11 and 11.02); its AOB decay is the measured 0.17 1/d
(the core's ASM2d autotrophs: 0.15).

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
    fluxes are integrated over 11 saved times, and the solve's rtol is 1e-8.
    The N balance counts the denitrified N leaving as stripped N2 (26.8% of
    the influent N at 20 C)."""
    _, _, result, solution = nitrite_bsm1
    assert bool(result["steady"].converged)
    balance = result["plant"].mass_balance(solution, components=("N", "P", "COD"))
    for q in ("N", "P", "COD"):
        assert abs(balance[q].imbalance) < 1e-9 * balance[q].inflow, q
    assert balance["N"].gas > 0.0


def test_level_nitrifies_through_nitrite_on_bsm1(nitrite_bsm1):
    b, model, result, _ = nitrite_bsm1
    tank5 = result["plant"].states_by_unit(result["steady"].state)["tank5"]

    def at(species):
        return float(tank5[model.species_index[species]])

    influent_nh4 = b.carry_over(model, b.BSM1_INFLUENT)["SNH4"]
    assert at("SNH4") < 0.1 * influent_nh4
    assert at("SNO2") < 0.5
    assert at("SNO3") > 10.0 * at("SNO2")
    assert at("XAOB") > at("XNOB") > 1.0
