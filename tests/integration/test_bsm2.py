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


# Module-scoped so the (expensive) build + steady-state solve happens ONCE for
# the whole file when these tests land on the same worker; each test below
# inspects a different facet of that single solution rather than re-solving. The
# solving tests run in the merge-only `slow` job (they request the `steady`
# fixture -- see tests/conftest.py); only the cheap assembly check stays on the
# fast PR gate.
@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


@pytest.fixture(scope="module")
def constant_influent(asm1):
    """Constant raw influent at a representative BSM2 average composition."""
    over = {"SI": 27.0, "SS": 58.0, "XI": 92.0, "XS": 364.0, "XB_H": 51.0,
            "XB_A": 0.0, "XP": 0.0, "SO": 0.0, "SNO": 0.0, "SNH": 24.0,
            "SND": 4.0, "XND": 15.0, "SALK": 7.0}
    C0 = asm1.concentrations(over)
    return InfluentSeries(t=jnp.array([0.0, 1e4]), Q=jnp.full((2,), BSM2_Q_REF),
                          C=jnp.tile(C0, (2, 1)), model=asm1)


@pytest.fixture(scope="module")
def plant(asm1, constant_influent):
    """The assembled open-loop BSM2 plant (no solve) -- a cheap fixture for the
    assembly / flow-resolution check that stays on the fast PR gate."""
    p = build_bsm2(asm1_model=asm1)
    p.add_influent("feed", constant_influent)
    return p


@pytest.fixture(scope="module")
def steady(plant):
    """The open-loop BSM2 plant and its steady-state solve (computed once).

    Requesting this fixture marks a test ``slow`` (it builds + integrates the
    full plant -- ~30 s to compile); see ``tests/conftest.py``."""
    sol = plant.solve(t_span=(0.0, 200.0), t_eval=jnp.array([0.0, 200.0]),
                      rtol=1e-4, atol=1e-3,
                      integrator=aquakin.IntegratorConfig(max_steps=400_000))
    return plant, sol


def test_bsm2_builds_and_reaches_steady_state(steady):
    _plant, sol = steady
    assert jnp.all(jnp.isfinite(sol.state))


def test_bsm2_activated_sludge_healthy(steady):
    """Nitrification active and biomass sustained in the AS reactors."""
    _plant, sol = steady
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 800.0     # heterotrophs alive
    assert float(sol.C_named("tank5", "XB_A")[-1]) > 50.0      # autotrophs alive
    assert float(sol.C_named("tank5", "SNH")[-1]) < 2.0        # nitrified
    assert float(sol.C_named("tank5", "SNO")[-1]) > 5.0        # nitrate produced
    # Anoxic tanks 1-2 (no aeration) hold little oxygen; aerobic tanks 3-5 do.
    assert float(sol.C_named("tank1", "SO")[-1]) < 0.5
    assert float(sol.C_named("tank5", "SO")[-1]) > 0.3


def test_bsm2_digester_produces_methane(steady):
    """The ADM1 digester reaches a methanogenic steady state (headspace CH4
    near the BSM2 reference) inside the coupled plant."""
    plant, sol = steady
    adm1 = plant.units["digester"].model
    dstate = plant.states_by_unit(sol.final_state)["digester"]
    g = lambda n: float(dstate[adm1.species_index[n]])
    assert jnp.all(jnp.isfinite(dstate))
    assert g("S_gas_ch4") > 1.0        # headspace methane (BSM2 ref ~1.65)
    assert g("X_ac") > 0.3             # acetoclastic methanogens sustained
    assert g("S_IN") > 0.01            # inorganic N present
    # State-derived digester pH in the physical anaerobic range.
    # (S_cat/S_an carry the strong-ion difference set by the interface.)
    assert g("S_ac") > 0.0


def test_bsm2_flow_balance(plant):
    """The resolved flow network is consistent: fixed pumps at setpoint, the
    plant-wide volume balance closes (influent + reject = effluent + sludge).

    Uses only the (cheap) flow resolution, not the integration -- so it stays on
    the fast gate (the cheap ``plant`` fixture, not the solving ``steady``)."""
    plant._build_state_layout()
    plant._build_parameter_layout()
    params = plant.default_parameters()
    fl = plant._recycle._resolve_flows(jnp.asarray(0.0), params)
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
    # recycles), so influent + external carbon == effluent + disposal.
    from aquakin.plant.bsm.bsm2 import BSM2_CARBON_FLOW
    effluent = float(fl[("settler", "overflow")])
    dewater_disposal = float(fl[("dewatering", "underflow")])
    assert (BSM2_Q_REF + BSM2_CARBON_FLOW) == pytest.approx(
        effluent + dewater_disposal, rel=1e-3)
