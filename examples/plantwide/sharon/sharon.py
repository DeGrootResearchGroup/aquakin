"""The plant-wide nitrite level in a SHARON reactor (Hellinga et al. 1998).

A stirred tank without sludge retention, so the sludge age is the residence
time (1.5 d), fed digester centrate (Hellinga et al.'s Table 1: 1000 g
NH4-N/m3, 4080 g HCO3-/m3, P 27 g/m3, BOD 230 g/m3). At 35 C their reactor
kept the AOB and washed the NOB out, so it oxidized ammonium to nitrite only;
at about 20 C neither group grows fast enough to stay (their Fig. 2).

Simplified against their lab reactor: continuous aeration, no methanol and so
no denitrification, and pH fixed at 7.4 (their predicted average; pH is a
condition at this level). Run as a script to print both temperatures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import aquakin
from aquakin.plant import Aeration, CSTRUnit, Plant
from aquakin.plant.influent import InfluentSeries

MODEL = Path(__file__).resolve().parents[1] / "models" / "plantwide_nitrite.yaml"
RESIDENCE_TIME = 1.5  # d
PH = 7.4
# Centrate (Hellinga et al. 1998, Table 1), in the level's species: HCO3- as
# inorganic carbon (4080 / 61 mol/m3 x 12 g C/mol), BOD as fermentable COD,
# the rest of the COD inert.
CENTRATE = dict(SNH4=1000.0, SIC=4080.0 / 61.0 * 12.0, SPO4=27.0, SF=230.0, SI=580.0)
# A start with both nitrifier groups present, so washout is decided by growth.
START = dict(SNH4=100.0, SIC=500.0, SPO4=20.0, SO2=2.0, XAOB=100.0, XNOB=100.0, XH=300.0)


def sharon_plant(model, T):
    """A 1.5 m3 tank fed 1 m3/d of centrate, aerated, at temperature ``T`` (K)."""
    plant = Plant("sharon", time_unit="d")
    plant.add_unit(
        CSTRUnit(
            name="reactor",
            model=model,
            volume=RESIDENCE_TIME,
            input_port_names=["inlet"],
            conditions={"T": T, "pH": PH},
            aeration=Aeration(kla=200.0, do_sat=7.0, species="SO2", stripped="model"),
        )
    )
    c = np.asarray(model.concentrations(CENTRATE, base="zero"))
    feed = InfluentSeries(t=np.array([0.0, 1e3]), Q=np.array([1.0, 1.0]), C=np.stack([c, c]), model=model)
    plant.add_influent("feed", feed, to="reactor.inlet")
    return plant


def run(model, T, days=60.0):
    """The reactor's ammonium, nitrite, nitrate, AOB and NOB at 0, ``days``/2
    and ``days`` (40 residence times by default)."""
    plant = sharon_plant(model, T)
    y0 = plant.initial_state(overrides={"reactor": model.concentrations(START, base="zero")})
    times = np.array([0.0, days / 2.0, days])
    sol = plant.solve(t_span=(0.0, days), t_eval=times, y0=y0, rtol=1e-6, atol=1e-6)
    species = ("SNH4", "SNO2", "SNO3", "XAOB", "XNOB")
    return {s: [float(v) for v in sol.C_named("reactor", s)] for s in species}


if __name__ == "__main__":
    nitrite = aquakin.load_model_from_file(MODEL)
    for t_c in (35.0, 20.0):
        out = run(nitrite, 273.15 + t_c)
        print(f"{t_c:.0f} C:", {s: round(v[-1], 2) for s, v in out.items()})
