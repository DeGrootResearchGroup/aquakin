"""The plant-wide nitrite level in a SHARON reactor (Hellinga et al. 1998).

A stirred tank without sludge retention (1.5 d residence time) fed digester
centrate with 1000 g NH4-N/m3 (``examples/plantwide/sharon``). Hellinga et
al.'s reactor at 35 C kept the AOB and washed the NOB out, oxidizing ammonium
to nitrite only; at typical plant temperatures neither group grows faster
than 1 / (1.5 d) (their Fig. 2). Checked, over 60 days (40 residence times)
from both groups at 100 g COD/m3:

- 35 C: the NOB wash out (below 1e-3 of the start), the AOB stay at a steady
  population, and the nitrogen they oxidize leaves as nitrite (nitrate below
  1e-3 of it);
- 20 C: the AOB wash out too (below 5% of the start, and still falling).

This needs the AOB not to be inhibited by free ammonia at the reactor's
roughly 10 g NH3-N/m3, as Hellinga et al. measured; with an inhibition
constant of 1 g N/m3 the AOB wash out at 35 C as well.

About 30 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import aquakin

pytestmark = pytest.mark.validation

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "plantwide" / "sharon"))


@pytest.fixture(scope="module")
def sharon():
    import sharon as s

    model = aquakin.load_model_from_file(s.MODEL)
    return {t_c: s.run(model, 273.15 + t_c) for t_c in (35.0, 20.0)}


def test_at_35_c_aob_stay_nob_wash_out_and_nitrite_is_the_product(sharon):
    """The AOB population at 30 and 60 days agrees to 1e-3 (steady); the
    reactor's nitrite is more than half the influent ammonium's 1000 g N/m3."""
    run = sharon[35.0]
    assert run["XNOB"][-1] < 1e-3 * run["XNOB"][0]
    assert run["XAOB"][-1] > 10.0
    assert run["XAOB"][-1] == pytest.approx(run["XAOB"][1], rel=1e-3)
    assert run["SNO2"][-1] > 500.0
    assert run["SNO3"][-1] < 1e-3 * run["SNO2"][-1]


def test_at_20_c_aob_wash_out_too(sharon):
    run = sharon[20.0]
    assert run["XAOB"][-1] < 0.05 * run["XAOB"][0]
    assert run["XAOB"][-1] < run["XAOB"][1]
    assert run["XNOB"][-1] < 1e-3 * run["XNOB"][0]
