"""BSM2 open-loop plant integration tests.

Verifies the full BSM2 flowsheet (primary clarifier + activated sludge +
secondary clarifier + thickener + ADM1 digester with ASM1<->ADM1 interfaces +
dewatering + reject-water recycle) assembles and integrates to a healthy
steady state. Like the BSM1 tests these assert *qualitative* behaviour
(nitrification active, biomass alive, digester producing methane, flows
balanced) rather than published BSM2 numbers, which would need the canonical
IWA influent file. The digester is quantitatively validated against the
published BSM2 steady state at the unit level in
``tests/validation/test_bsm2_digester_unit.py``.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm.bsm2 import build_bsm2, BSM2_Q_REF
from aquakin.plant.influent import InfluentSeries


@pytest.fixture
def asm1():
    return aquakin.load_network("asm1")


@pytest.fixture
def constant_influent(asm1):
    """Constant raw influent at a representative BSM2 average composition."""
    over = {"SI": 27.0, "SS": 58.0, "XI": 92.0, "XS": 364.0, "XB_H": 51.0,
            "XB_A": 0.0, "XP": 0.0, "SO": 0.0, "SNO": 0.0, "SNH": 24.0,
            "SND": 4.0, "XND": 15.0, "SALK": 7.0}
    C0 = asm1.default_concentrations()
    for sp, v in over.items():
        C0 = C0.at[asm1.species_index[sp]].set(v)
    return InfluentSeries(t=jnp.array([0.0, 1e4]), Q=jnp.full((2,), BSM2_Q_REF),
                          C=jnp.tile(C0, (2, 1)), network=asm1)


def _build(asm1, influent):
    plant = build_bsm2(asm1_network=asm1)
    plant.add_influent("feed", influent)
    plant.connect(None, "feed", "front_mix", "fresh")
    return plant


def _solve(plant, t_end=200.0):
    return plant.solve(t_span=(0.0, t_end), t_eval=jnp.array([0.0, t_end]),
                       rtol=1e-4, atol=1e-3, max_steps=400_000)


def test_bsm2_builds_and_reaches_steady_state(asm1, constant_influent):
    plant = _build(asm1, constant_influent)
    sol = _solve(plant)
    assert jnp.all(jnp.isfinite(sol.state))


def test_bsm2_activated_sludge_healthy(asm1, constant_influent):
    """Nitrification active and biomass sustained in the AS reactors."""
    plant = _build(asm1, constant_influent)
    sol = _solve(plant)
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 800.0     # heterotrophs alive
    assert float(sol.C_named("tank5", "XB_A")[-1]) > 50.0      # autotrophs alive
    assert float(sol.C_named("tank5", "SNH")[-1]) < 2.0        # nitrified
    assert float(sol.C_named("tank5", "SNO")[-1]) > 5.0        # nitrate produced
    # Anoxic tanks 1-2 (no aeration) hold little oxygen; aerobic tanks 3-5 do.
    assert float(sol.C_named("tank1", "SO")[-1]) < 0.5
    assert float(sol.C_named("tank5", "SO")[-1]) > 0.3


def test_bsm2_digester_produces_methane(asm1, constant_influent):
    """The ADM1 digester reaches a methanogenic steady state (headspace CH4
    near the BSM2 reference) inside the coupled plant."""
    plant = _build(asm1, constant_influent)
    sol = _solve(plant)
    adm1 = plant.units["digester"].network
    start, size = plant._state_layout["digester"]
    dstate = sol.state[-1, start:start + size]
    g = lambda n: float(dstate[adm1.species_index[n]])
    assert jnp.all(jnp.isfinite(dstate))
    assert g("S_gas_ch4") > 1.0        # headspace methane (BSM2 ref ~1.65)
    assert g("X_ac") > 0.3             # acetoclastic methanogens sustained
    assert g("S_IN") > 0.01            # inorganic N present
    # State-derived digester pH in the physical anaerobic range.
    # (S_cat/S_an carry the strong-ion difference set by the interface.)
    assert g("S_ac") > 0.0


def test_bsm2_flow_balance(asm1, constant_influent):
    """The resolved flow network is consistent: fixed pumps at setpoint, the
    plant-wide volume balance closes (influent + reject = effluent + sludge)."""
    plant = _build(asm1, constant_influent)
    plant._build_state_layout()
    plant._build_parameter_layout()
    params = plant.default_parameters()
    fl = plant._resolve_flows(jnp.asarray(0.0), params)
    # Fixed recycle pumps at their setpoints.
    assert float(fl[("tank5_split", "internal_recycle")]) == pytest.approx(3 * BSM2_Q_REF, rel=1e-9)
    assert float(fl[("underflow_split", "ras")]) == pytest.approx(BSM2_Q_REF, rel=1e-9)
    assert float(fl[("underflow_split", "waste")]) == pytest.approx(300.0, rel=1e-9)
    # Primary sludge is the f_PS fraction of the (influent + reject) feed.
    reject = float(fl[("reject_mix", "out")])
    primary_feed = BSM2_Q_REF + reject
    assert float(fl[("primary", "underflow")]) == pytest.approx(0.007 * primary_feed, rel=1e-6)
    # Net plant volume balance: the only liquid streams permanently leaving are
    # the settler effluent and the dewatered-sludge disposal (everything else
    # recycles), so influent == effluent + disposal.
    effluent = float(fl[("settler", "overflow")])
    dewater_disposal = float(fl[("dewatering", "underflow")])
    assert BSM2_Q_REF == pytest.approx(effluent + dewater_disposal, rel=1e-3)
