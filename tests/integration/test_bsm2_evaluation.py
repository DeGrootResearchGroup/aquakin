"""BSM2 performance-evaluation tests (EQI / OCI).

Exercises ``evaluate_bsm2``: the wiring from a solved BSM2 plant to the
headline Effluent Quality Index and Operational Cost Index. One open-loop solve
is shared at module scope (the suite runs close to the CI runner's limit, so the
heavy full-plant solves are kept to a minimum); the open-vs-closed-loop
comparison that the control work motivates is demonstrated in
``examples/bsm2_evaluation.py``.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import (
    BSM2Evaluation,
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
    evaluate_bsm2,
)


@pytest.fixture(scope="module")
def asm1():
    return bsm2_asm1_network()


@pytest.fixture(scope="module")
def adm1():
    return aquakin.load_network("adm1")


@pytest.fixture(scope="module")
def evaluated(asm1, adm1):
    """Build an open-loop BSM2 plant, solve briefly, and evaluate it once."""
    plant = build_bsm2(asm1, adm1, do_control=False)
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")
    params = bsm2_parameters(asm1, adm1)
    sol = plant.solve((0.0, 10.0), t_eval=jnp.linspace(0.0, 10.0, 11),
                      params=params, rtol=1e-4, atol=1e-3, max_steps=300_000)
    return plant, sol, params, evaluate_bsm2(plant, sol, params)


def test_evaluation_terms_finite_and_positive(evaluated):
    _, _, _, ev = evaluated
    assert isinstance(ev, BSM2Evaluation)
    # Strictly positive terms.
    for name, val in (("eqi", ev.eqi),
                      ("aeration_energy", ev.aeration_energy),
                      ("pumping_energy", ev.pumping_energy),
                      ("mixing_energy", ev.mixing_energy),
                      ("sludge_production", ev.sludge_production),
                      ("carbon_mass", ev.carbon_mass),
                      ("methane_production", ev.methane_production)):
        assert jnp.isfinite(val), f"{name} not finite: {val}"
        assert val > 0.0, f"{name} not positive: {val}"
    # Non-negative / finite terms.
    assert jnp.isfinite(ev.heating_energy) and ev.heating_energy >= 0.0
    assert jnp.isfinite(ev.oci)


def test_aerated_tanks_detected(evaluated):
    """The three aerobic reactors are the ones whose aeration is counted."""
    _, _, _, ev = evaluated
    assert ev.aerated_tanks == ["tank3", "tank4", "tank5"]


def test_aeration_energy_matches_fixed_kla(evaluated, asm1, adm1):
    """For the open-loop plant kLa is fixed, so the aeration energy equals the
    closed-form ``S_sat / (1.8e3) * Σ_i V_i kLa_i`` independent of the horizon."""
    from aquakin.plant.bsm.bsm2 import (
        BSM2_DO_SATURATION, BSM2_KLA, BSM2_TANK_VOLUMES,
    )
    _, _, _, ev = evaluated
    expected = BSM2_DO_SATURATION / 1800.0 * sum(
        BSM2_TANK_VOLUMES[i] * BSM2_KLA[i] for i in range(5))
    assert ev.aeration_energy == pytest.approx(expected, rel=1e-6)


def test_mixing_energy_counts_anoxic_reactors_and_digester(evaluated):
    """Open-loop: only the anoxic reactors (kLa=0) and the always-mixed digester
    contribute; ``ME = 24 * 0.005 * (V_anoxic + V_digester)``."""
    from aquakin.plant.bsm.bsm2 import BSM2_TANK_VOLUMES, BSM2_DIGESTER_VOLUME
    _, _, _, ev = evaluated
    v_anoxic = BSM2_TANK_VOLUMES[0] + BSM2_TANK_VOLUMES[1]  # tanks 1, 2
    expected = 24.0 * 0.005 * (v_anoxic + BSM2_DIGESTER_VOLUME)
    assert ev.mixing_energy == pytest.approx(expected, rel=1e-6)


def test_carbon_mass_matches_dose(evaluated):
    """Carbon mass = carbon flow x source concentration (kg COD/d)."""
    from aquakin.plant.bsm.bsm2 import BSM2_CARBON_FLOW, BSM2_CARBON_CONC
    _, _, _, ev = evaluated
    assert ev.carbon_mass == pytest.approx(
        BSM2_CARBON_FLOW * BSM2_CARBON_CONC / 1000.0, rel=1e-6)


def test_oci_is_full_bsm2_sum(evaluated):
    """OCI = AE + PE + ME + 3*sludge + 3*carbon - 6*methane
            + max(0, HE - 7*methane) (Gernaey et al. 2014)."""
    _, _, _, ev = evaluated
    expected = (ev.aeration_energy + ev.pumping_energy + ev.mixing_energy
                + 3.0 * ev.sludge_production + 3.0 * ev.carbon_mass
                - 6.0 * ev.methane_production
                + max(0.0, ev.heating_energy - 7.0 * ev.methane_production))
    assert ev.oci == pytest.approx(expected, rel=1e-9)


def test_eqi_consistent_with_effluent(evaluated):
    """A nitrifying plant has low effluent ammonia and a positive EQI dominated
    by the nitrate term -- a sanity bound rather than a published value."""
    _, _, _, ev = evaluated
    assert ev.effluent["SNH"] < 5.0      # nitrified
    assert ev.effluent["SNO"] > 1.0      # nitrate present
    assert ev.eqi > 0.0


def test_signals_at_open_loop_is_empty(evaluated):
    """An open-loop plant publishes no control signals."""
    plant, sol, params, _ = evaluated
    assert plant.signals_at(sol.t[0], sol.state[0], params) == {}


# ----- OCI-component metric kernels (no plant solve) ----------------------

def test_mixing_energy_kernel():
    """Only reactors below the kLa threshold (plus the digester) are mixed; the
    energy is the time-fraction-weighted volume times 24*unit."""
    from aquakin.plant.metrics import mixing_energy
    t = jnp.array([0.0, 1.0])
    # Two reactors: one always unaerated (kLa=0), one always aerated (kLa=100).
    kla = jnp.array([[0.0, 100.0], [0.0, 100.0]])
    volumes = jnp.array([1500.0, 3000.0])
    me = mixing_energy(t, kla, volumes, digester_volume=3400.0)
    expected = 24.0 * 0.005 * (1500.0 + 3400.0)  # aerated reactor excluded
    assert me == pytest.approx(expected, rel=1e-9)


def test_pumping_energy_bsm2_kernel():
    from aquakin.plant.metrics import pumping_energy_bsm2
    t = jnp.array([0.0, 1.0])
    flows = {"internal": jnp.array([100.0, 100.0]),
             "ras": jnp.array([50.0, 50.0]),
             "dewatering_underflow": jnp.array([10.0, 10.0])}
    pe = pumping_energy_bsm2(t, flows)
    assert pe == pytest.approx(0.004 * 100 + 0.008 * 50 + 0.004 * 10, rel=1e-9)


def test_carbon_mass_kernel():
    from aquakin.plant.metrics import carbon_mass
    t = jnp.array([0.0, 1.0])
    cm = carbon_mass(t, jnp.array([2.0, 2.0]), carbon_conc=400000.0)
    assert cm == pytest.approx(2.0 * 400000.0 / 1000.0, rel=1e-9)  # 800 kg COD/d


def test_heating_energy_kernel():
    from aquakin.plant.metrics import heating_energy
    t = jnp.array([0.0, 1.0])
    Q = jnp.array([100.0, 100.0])
    he = heating_energy(t, Q, T_feed_C=15.0, T_target_C=35.0)
    expected = 24.0 * (35.0 - 15.0) * 100.0 * 1000.0 * 4.186 / 86400.0
    assert he == pytest.approx(expected, rel=1e-9)


def test_oci_bsm2_methane_offsets_heating():
    """When methane covers the heating demand, the heating term contributes 0."""
    from aquakin.plant.metrics import operational_cost_index_bsm2
    # heating - 7*methane = 100 - 7*50 < 0 -> max(0, ...) = 0.
    oci = operational_cost_index_bsm2(
        aeration=10, pumping=5, mixing=3, sludge_production=2,
        carbon=1, methane=50, heating=100)
    assert oci == pytest.approx(10 + 5 + 3 + 3 * 2 + 3 * 1 - 6 * 50 + 0.0, rel=1e-9)
