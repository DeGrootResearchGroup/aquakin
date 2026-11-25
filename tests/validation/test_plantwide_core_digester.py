"""The plant-wide core model's digester against BSM2's ADM1 digester.

Both digest the same sludge: the BSM2 open-loop steady state's digester feed,
the activated-sludge (ASM1) sludge mix of primary and thickened secondary
sludge. ADM1 sees it through BSM2's ASM1-to-ADM1 interface; the core model,
whose species are the activated-sludge ones, takes it directly. The reference
is the published BSM2 digester steady state (``REFERENCE_SS``), and the
digester has BSM2's dimensions and temperature.

Checked:

- the feed is BSM2's: put through the interface, it reproduces the published
  ADM1 digester input;
- the core digester's methane production and particulate COD destruction
  equal ADM1's at BSM2's steady state (the heterotroph anaerobic hydrolysis
  factor ``etafe`` = 0.15 was chosen by this comparison).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import Plant
from aquakin.plant.headspace import HeadspaceDigesterUnit
from aquakin.plant.influent import InfluentSeries
from aquakin.plant.interfaces import ASM1toADM1
from tests.validation.bsm2_digester_reference import INFLUENT, REFERENCE_SS, T_OP, V_LIQ

pytestmark = pytest.mark.validation

CORE = Path(__file__).parents[2] / "examples" / "plantwide" / "models" / "plantwide_core.yaml"

# The digester feed (ASM1, g/m3; SALK mol/m3) of aquakin's BSM2 at its
# constant-influent open-loop steady state (``build_bsm2`` with
# ``bsm2_parameters``, solved by ``plant.steady_state`` from
# ``bsm2_warm_start``): the ``sludge_mix`` output.
Q_FEED = 178.3695
ASM1_FEED = {
    "SI": 28.0656, "SS": 48.9797, "XI": 10366.0329, "XS": 20385.7953,
    "XB_H": 10158.5609, "XB_A": 549.8798, "XP": 3218.0546, "SO": 0.2507,
    "SNO": 1.6704, "SNH": 28.9021, "SND": 4.6861, "XND": 906.5724, "SALK": 7.1548,
}  # fmt: skip
# BSM2's digester pH at its steady state; pH is a condition of the core model.
PH = 7.26
# Particulate organics of each model, g COD/m3 (ADM1 states are kg/m3).
CORE_PARTICULATES = ("XI", "XS", "XH", "XAUT", "XFE", "XPRO", "XAC", "XH2M")
ADM1_PARTICULATES = (
    "X_c", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
)  # fmt: skip


def test_feed_translates_to_the_published_bsm2_digester_input():
    """Through BSM2's interface the feed is the published ADM1 input, to 0.3%
    on every organic and nitrogen state (the inorganic carbon depends on the
    interface's charge balance and is not compared)."""
    asm1, adm1 = aquakin.load_model("asm1"), aquakin.load_model("adm1")
    C = jnp.array([ASM1_FEED[s] for s in asm1.species])
    out = ASM1toADM1(source_model=asm1, target_model=adm1).translate(C)
    for name, ref in INFLUENT.items():
        if name == "S_IC":
            continue
        assert float(out[adm1.species_index[name]]) == pytest.approx(ref, rel=3e-3), name


def _core_feed(core):
    """The ASM1 feed in the core model's species. Inert particulates and
    endogenous products are both ``XI``; alkalinity is inorganic carbon. The
    models' organic N contents differ (ASM1 0.08 g N/g COD in biomass and 0.06
    in inerts), so total N is kept equal by putting the difference in
    ammonium."""
    a = ASM1_FEED
    feed = {
        "SF": a["SS"], "SI": a["SI"], "XS": a["XS"], "XH": a["XB_H"], "XAUT": a["XB_A"],
        "XI": a["XI"] + a["XP"], "SO2": a["SO"], "SNO3": a["SNO"], "SIC": a["SALK"] * 12.0,
    }  # fmt: skip
    n_fed = (
        a["SNH"] + a["SND"] + a["XND"] + a["SNO"]
        + 0.08 * (a["XB_H"] + a["XB_A"]) + 0.06 * (a["XI"] + a["XP"])
    )  # fmt: skip
    comp = core.composition()
    n_organic = sum(comp.get(s, {}).get("N", 0.0) * v for s, v in feed.items())
    feed["SNH4"] = n_fed - n_organic
    return feed


@pytest.fixture(scope="module")
def core_digester():
    """The core digester on the BSM2 feed, after 900 days (steady: over the
    last 100 days no state changes by more than the solve's rtol, 1e-6)."""
    core = aquakin.load_model_from_file(CORE)
    feed = _core_feed(core)

    def vector(values):
        return jnp.array([values.get(s, 0.0) for s in core.species])

    plant = Plant("digester", time_unit="d")
    plant.add_unit(
        HeadspaceDigesterUnit(
            name="digester",
            model=core,
            volume=V_LIQ,
            gas_volume=300.0,
            conditions={"T": T_OP, "pH": PH},
        )
    )
    C_in = vector(feed)
    plant.add_influent(
        "sludge",
        InfluentSeries(
            t=jnp.asarray([0.0, 1000.0]),
            Q=jnp.asarray([Q_FEED, Q_FEED]),
            C=jnp.stack([C_in, C_in]),
            model=core,
        ),
        to="digester.inlet",
    )
    seed = {**feed, "XFE": 500.0, "XPRO": 100.0, "XAC": 500.0, "XH2M": 200.0}
    y0 = plant.initial_state(
        overrides={"digester": jnp.concatenate([vector(seed), jnp.zeros(len(core.gases.entries))])}
    )
    sol = plant.solve(t_span=(0.0, 900.0), t_eval=jnp.asarray([800.0, 900.0]), y0=y0)
    return core, feed, plant, sol


def _adm1_methane():
    """Methane production (kg CH4/d) of the published BSM2 steady state, by the
    BSM2 evaluation formula and the ADM1 gas parameters."""
    adm1 = aquakin.load_model("adm1")
    p = dict(zip(adm1.parameters, np.asarray(adm1.default_parameters())))
    R_T, P_atm = p["R_T"], p["P_atm"]
    p_ch4 = R_T / 64.0 * REFERENCE_SS["S_gas_ch4"]
    P = R_T / 16.0 * REFERENCE_SS["S_gas_h2"] + p_ch4 + R_T * REFERENCE_SS["S_gas_co2"]
    P = P + p["p_h2o"]
    Q_gas = p["k_P"] * (P - P_atm) * P / P_atm
    return (p_ch4 / P) * P_atm * 16.0 / R_T * Q_gas


def test_core_digester_methane_and_solids_destruction_equal_adm1(core_digester):
    """Methane to 0.5% (1064 against 1065 kg CH4/d) and particulate COD
    destroyed to one percentage point (53.8% against 54.2%). Both models are
    fed the same particulate COD, so the destruction compares directly.

    The tolerances are the model difference, not solver error: the two models
    reach these totals by different routes (the core treats 90% of decaying
    biomass as biodegradable where BSM2's interface treats 68%, and leaves more
    of it unhydrolyzed), so they agree on totals, not on every state."""
    core, feed, plant, sol = core_digester
    state = np.asarray(sol.unit_state("digester"))
    n = core.n_species
    assert np.max(np.abs(state[-1, :n] - state[-2, :n]) / (np.abs(state[-1, :n]) + 1.0)) < 1e-6

    methane = float(plant.digester_gas(sol).ch4[-1])
    assert methane == pytest.approx(_adm1_methane(), rel=5e-3)

    fed = sum(feed.get(s, 0.0) for s in CORE_PARTICULATES)
    left = sum(state[-1, core.species_index[s]] for s in CORE_PARTICULATES)
    adm1_left = 1000.0 * sum(REFERENCE_SS[s] for s in ADM1_PARTICULATES)
    assert 1.0 - left / fed == pytest.approx(1.0 - adm1_left / fed, abs=0.01)
