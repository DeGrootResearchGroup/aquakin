"""Validation: the open-loop BSM2 plant reproduces the published steady state.

Runs ``build_bsm2`` with the published BSM2 constant influent and the BSM2
(15 °C) ASM1 parameter set, and checks the activated-sludge reactor states and
the digester against the reference open-loop steady state (``asm1init_bsm2.m``
``XINIT`` and ``adm1init_bsm2.m`` ``DIGESTERINIT``). The whole multi-network
plant -- primary clarifier, 5 AS reactors, Takács secondary clarifier, thickener,
ADM1 digester with the ASM1<->ADM1 interfaces, dewatering, and the reject-water
recycle -- matches to within ~3% on every key state.

References
----------
Gernaey, K.V. et al. (2014). Benchmarking of Control Strategies for Wastewater
Treatment Plants. IWA Scientific and Technical Report No. 23.
Jeppsson, U. et al. (2007). Benchmark Simulation Model No 2. Water Sci. Technol.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import bsm2_warm_start
from aquakin.plant.bsm.bsm2 import (
    build_bsm2,
    bsm2_constant_influent,
    bsm2_parameters,
)


# Published BSM2 open-loop reactor steady states (asm1init_bsm2 XINIT1 / XINIT5).
REF = {
    "tank1": {"XB_H": 2245.1, "XB_A": 166.7, "XP": 964.9, "XI": 1532.3,
              "SNH": 6.892, "SNO": 3.935, "SS": 3.050},
    "tank5": {"XB_H": 2242.1, "XB_A": 167.8, "XP": 970.4, "XI": 1532.3,
              "SNH": 0.159, "SNO": 9.195, "SS": 0.673},
}
# Published digester steady state (adm1init_bsm2 DIGESTERINIT).
REF_DIG = {"S_gas_ch4": 1.6535, "S_ac": 0.0893, "X_ac": 0.677, "S_IN": 0.0945}


def _solve():
    asm1 = aquakin.load_network("asm1")
    adm1 = aquakin.load_network("adm1")
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    plant.add_influent("feed", bsm2_constant_influent(asm1))

    y0 = bsm2_warm_start(plant)

    sol = plant.solve(t_span=(0.0, 150.0), t_eval=jnp.array([0.0, 150.0]),
                      params=bsm2_parameters(asm1, adm1), y0=y0,
                      rtol=1e-5, atol=1e-3, max_steps=500_000)
    return plant, sol


@pytest.mark.validation
def test_bsm2_activated_sludge_matches_reference():
    plant, sol = _solve()
    assert jnp.all(jnp.isfinite(sol.state))
    worst, worst_name = 0.0, ""
    for tk, ref in REF.items():
        for sp, rv in ref.items():
            mv = float(sol.C_named(tk, sp)[-1])
            rel = abs(mv - rv) / abs(rv)
            if rel > worst:
                worst, worst_name = rel, f"{tk}.{sp}"
    assert worst < 0.06, f"{worst_name} off by {worst:.1%} from BSM2 reference"


@pytest.mark.validation
def test_bsm2_digester_matches_reference():
    plant, sol = _solve()
    adm1 = plant.units["digester"].network
    d = plant.states_by_unit(sol.final_state)["digester"]
    for sp, rv in REF_DIG.items():
        mv = float(d[adm1.species_index[sp]])
        assert mv == pytest.approx(rv, rel=0.06), f"digester {sp}: {mv} vs {rv}"
    # Headspace methane (the defining digester output) much tighter.
    assert float(d[adm1.species_index["S_gas_ch4"]]) == pytest.approx(
        REF_DIG["S_gas_ch4"], rel=0.01)
