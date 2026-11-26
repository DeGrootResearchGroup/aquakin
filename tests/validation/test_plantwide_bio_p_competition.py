"""The plant-wide bio-P level reproduces the PAO-GAO competition of
López-Vázquez et al. (2009).

The generated model (``plantwide_bio_p``) runs, through its own reaction term,
in the source's sequencing batch reactor (reconstructed: 6 h cycle, 2 h
anaerobic, 3.5 h aerobic, 50% exchange, 8-day sludge age, 5 C-mmol/L VFA,
50 days) on four of the source's 36 conditions, one per regime. Checked:

- the dominant organism is the source's (its Fig. 6: dominant at >= 60% of
  the biomass, otherwise the two largest coexist);
- the population shares equal those of the faithful reference implementation
  of the source (``examples/plantwide/lopez_vazquez_2009``, PHA by type, yields
  from the current composition) to 3 points: the most the model's
  simplification (PHA pools by origin, constant yields) was measured to cost
  over the whole grid is 3.1, and at these four it is 1.2.

The whole grid (about 25 minutes) is ``aquakin_grid.py`` in that folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.validation

REFERENCE = Path(__file__).parents[2] / "examples" / "plantwide" / "lopez_vazquez_2009"
sys.path.insert(0, str(REFERENCE))

# (feed acetate fraction, T in C, pH): (the source's Fig. 6 outcome, the
# faithful reference's PAO / Competibacter / Alphaproteobacteria shares).
CASES = {
    (1.0, 10, 7.0): ("PAO", (0.99, 0.01, 0.00)),
    (1.0, 20, 7.0): ("Competi+PAO", (0.53, 0.46, 0.01)),
    (0.75, 20, 7.0): ("PAO", (0.90, 0.05, 0.04)),
    (0.0, 30, 7.0): ("Alpha", (0.18, 0.00, 0.82)),
}


@pytest.mark.parametrize("case", list(CASES), ids=lambda c: f"HAc{c[0]}-{c[1]}C-pH{c[2]}")
def test_competition_matches_the_source(case):
    import aquakin_grid
    from grid import classify

    f_hac, T, pH = case
    outcome, reference = CASES[case]
    shares = aquakin_grid.run(T, pH, f_hac)
    assert classify(shares) == outcome
    for pop, ref in zip(("PAO", "CMP", "ALP"), reference):
        assert shares[pop] == pytest.approx(ref, abs=0.03), pop
