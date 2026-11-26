"""The plant-wide bio-P level (``plantwide_bio_p``) on an A2O, against ASM2d.

Both models run on ``build_a2o`` with ASM2d's default influent, carried over
species by species (``examples/plantwide/a2o``); the plant-wide model's VFA is
75% acetate and 25% propionate. Checked:

- the plant-wide steady state is stable: both GAO groups, seeded at 1 g COD/m3
  into it, die out over 150 days (an equilibrium without GAOs exists whether
  or not they could invade, so convergence alone does not show this);
- phosphorus is conserved over the plant: over the last 75 days, what the
  influent brings equals what the effluent and wastage carry out plus what the
  plant accumulates;
- biological P removal as ASM2d gives it on the same plant: the anaerobic zone
  ferments its fermentable substrate, the PAOs release phosphate there and
  take it up in the aerobic zone, and the effluent phosphate is within
  1 g P/m3 of ASM2d's.

About 4 minutes: the steady states, then the 150-day stability run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.validation

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "plantwide" / "a2o"))


@pytest.fixture(scope="module")
def runs():
    import a2o_comparison

    plantwide = a2o_comparison.run_plantwide()
    asm2d = a2o_comparison.run_asm2d()
    return a2o_comparison, plantwide, asm2d


def test_steady_state_converges_and_gaos_cannot_invade_it(runs):
    _, plantwide, _ = runs
    assert bool(plantwide["steady"].converged)
    for group, rate in plantwide["gao_growth"].items():
        assert rate < -0.01, group


def test_phosphorus_is_conserved_over_the_plant(runs):
    """Influent P = effluent and wastage P + accumulation, to 1e-4 of the
    inflow: the solve's rtol is 1e-6, and the fluxes are integrated by the
    trapezoid rule over 16 saved times on an almost constant state."""
    _, plantwide, _ = runs
    balance = plantwide["plant"].mass_balance(plantwide["solution"], components=("P",))
    p = balance["P"]
    assert abs(p.imbalance) < 1e-4 * p.inflow


def test_biological_p_removal_matches_asm2d(runs):
    """ASM2d on this plant: SF 0.93 g/m3 left at the end of the anaerobic zone,
    effluent phosphate 5.13 g P/m3. The models differ in what decides the
    effluent phosphate (GAOs, P released per VFA, fermentation as growth), so
    it is compared to 1 g P/m3, and the fermentation by its result: the
    anaerobic zone leaves no more SF than ASM2d's."""
    module, plantwide, asm2d = runs
    pw, ref = module.summary(plantwide), module.summary(asm2d)
    p_in = module.plantwide_influent(plantwide["model"])["SPO4"]
    assert pw["SF_anaerobic"] <= ref["SF_anaerobic"]
    assert pw["PO4_anaerobic"] > 2.0 * p_in
    assert pw["PO4_effluent"] < 0.8 * p_in
    assert pw["PO4_effluent"] == pytest.approx(ref["PO4_effluent"], abs=1.0)
