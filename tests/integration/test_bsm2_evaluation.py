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
    plant.add_influent("feed", bsm2_constant_influent(asm1))
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


# ----- single-point (steady-state) degeneracy (issue #180) -----------------
# run_to_steady_state returns a one-point solution; every time-averaged kernel
# must then return the INSTANTANEOUS value (the average of a constant), not crash
# (ZeroDivisionError, the bug) nor a spurious zero.

def test_aeration_energy_single_point_is_instantaneous():
    from aquakin.plant.metrics import aeration_energy
    t1 = jnp.array([5.0])                     # one saved time
    kla = jnp.array([[240.0, 84.0]])          # (1, 2)
    volumes = jnp.array([1333.0, 1333.0])
    ae = aeration_energy(t1, kla, volumes, saturation=8.0)
    expected = 8.0 / 1800.0 * (240.0 * 1333.0 + 84.0 * 1333.0)
    assert ae == pytest.approx(expected, rel=1e-9)
    # ... and equals the multi-point limit of a constant trajectory.
    t2 = jnp.array([5.0, 6.0])
    ae2 = aeration_energy(t2, jnp.tile(kla, (2, 1)), volumes, saturation=8.0)
    assert ae == pytest.approx(ae2, rel=1e-9)


def test_time_averaged_kernels_single_point_dont_crash():
    from aquakin.plant.metrics import (
        carbon_mass, heating_energy, mixing_energy, pumping_energy,
        pumping_energy_bsm2,
    )
    t1 = jnp.array([3.0])
    assert pumping_energy(t1, jnp.array([100.0]), jnp.array([50.0]),
                          jnp.array([10.0])) == pytest.approx(
        0.004 * 100 + 0.008 * 50 + 0.05 * 10, rel=1e-9)
    assert pumping_energy_bsm2(t1, {"internal": jnp.array([100.0])}) == \
        pytest.approx(0.004 * 100, rel=1e-9)
    assert carbon_mass(t1, jnp.array([2.0]), carbon_conc=400000.0) == \
        pytest.approx(2.0 * 400000.0 / 1000.0, rel=1e-9)   # 800 kg COD/d
    he = heating_energy(t1, jnp.array([100.0]), T_feed_C=15.0, T_target_C=35.0)
    assert he == pytest.approx(24.0 * 20.0 * 100.0 * 1000.0 * 4.186 / 86400.0,
                               rel=1e-9)
    # One reactor unaerated, one aerated, at a single instant.
    me = mixing_energy(t1, jnp.array([[0.0, 100.0]]), jnp.array([1500.0, 3000.0]),
                       digester_volume=3400.0)
    assert me == pytest.approx(24.0 * 0.005 * (1500.0 + 3400.0), rel=1e-9)


def test_eqi_single_point_is_instantaneous(asm1):
    from aquakin.plant.metrics import effluent_quality_index
    C0 = asm1.concentrations({"XS": 10.0, "SS": 5.0, "SNO": 8.0, "SNH": 2.0})
    C = C0[None, :]                            # (1, n_species)
    t1 = jnp.array([4.0])
    Q = jnp.array([18000.0])
    eqi1 = effluent_quality_index(t1, C, Q, asm1)
    # Equals the constant-trajectory two-point average.
    eqi2 = effluent_quality_index(jnp.array([4.0, 5.0]),
                                  jnp.tile(C, (2, 1)), jnp.tile(Q, 2), asm1)
    assert jnp.isfinite(eqi1) and eqi1 > 0.0
    assert eqi1 == pytest.approx(eqi2, rel=1e-9)


def test_effluent_averages_single_point(asm1):
    from aquakin.plant.metrics import effluent_averages
    C0 = asm1.concentrations({"XS": 10.0, "SS": 5.0, "SNO": 8.0, "SNH": 2.0})
    avg = effluent_averages(jnp.array([4.0]), C0[None, :],
                            jnp.array([18000.0]), asm1)
    assert avg["SNH"] == pytest.approx(2.0, rel=1e-9)
    assert avg["SNO"] == pytest.approx(8.0, rel=1e-9)
    assert all(jnp.isfinite(v) for v in avg.values())


# --- labeled EQI/OCI report (#153) -- fast, no solve --------------------------

_EFF = {"COD": 48.2, "BOD": 2.7, "TSS": 12.5, "TKN": 4.6, "SNH": 1.2, "SNO": 8.9}


def test_bsm2_report_is_labeled_with_units_and_breakdown():
    from aquakin.plant.bsm import BSM2Evaluation
    ev = BSM2Evaluation(
        eqi=6123.4, oci=0.0, aeration_energy=3784.2, pumping_energy=1689.0,
        mixing_energy=768.0, sludge_production=2280.5, carbon_mass=800.0,
        methane_production=1010.3, heating_energy=4200.0, effluent=_EFF,
        aerated_tanks=["tank3", "tank4", "tank5"])
    r = ev.report()
    assert str(ev) == r                                    # __str__ delegates
    # headline labels + units
    for token in ("BSM2 performance indices", "EQI", "OCI", "kg poll.-units/d",
                  "kWh/d", "kg TSS/d", "kg COD/d", "kg CH4/d", "g COD/m³"):
        assert token in r, token
    # the OCI formula and the caveat (oci_note) are always shown
    assert "AE + PE + ME + 3*sludge" in r
    assert "Note:" in r and "Gernaey" in r
    # the methane term shows its negative (credit) contribution
    assert f"{-6.0 * 1010.3:12.1f}".strip() in r
    # every aerated reactor is named
    assert "tank3, tank4, tank5" in r


def test_bsm1_report_is_labeled_and_str_delegates():
    from aquakin.plant.bsm import BSM1Evaluation
    ev = BSM1Evaluation(
        eqi=6443.2, oci=3341.4 + 388.2 + 5.0 * 2082.3, aeration_energy=3341.4,
        pumping_energy=388.2, sludge_production=2082.3, effluent=_EFF,
        aerated_tanks=["tank3", "tank4", "tank5"])
    r = str(ev)
    assert "BSM1 performance indices" in r
    assert "AE + PE + 5*sludge" in r and "Copp 2002" in r
    assert "kWh/d" in r and "kg TSS/d" in r and "kg poll.-units/d" in r
    # the OCI equals the sum of the displayed contributions
    assert ev.oci == pytest.approx(
        ev.aeration_energy + ev.pumping_energy + 5.0 * ev.sludge_production)


def test_eqi_weights_are_copp_alex_standard(asm1):
    """EQI uses the Copp 2002 / Alex 2008 weights (TSS 2, COD 1, BOD 2,
    TKN 30, NO 10). Guards against a regression in the nitrogen weighting,
    which otherwise leaves the effluent concentrations correct while the
    aggregate index is wrong."""
    from aquakin.plant.metrics import (
        effluent_quality_index, derived_TSS, derived_COD, derived_BOD,
        derived_TKN)
    C = asm1.concentrations({"SI": 28.0, "SS": 5.0, "XI": 10.0, "XS": 8.0,
                             "XB_H": 12.0, "SNH": 2.0, "SNO": 7.0, "SND": 1.0,
                             "XND": 1.5})
    Ctraj = jnp.stack([C, C])
    t = jnp.array([0.0, 1.0])
    Q = jnp.array([1.0e4, 1.0e4])
    tss = float(derived_TSS(Ctraj, asm1)[0])
    cod = float(derived_COD(Ctraj, asm1)[0])
    bod = float(derived_BOD(Ctraj, asm1)[0])
    tkn = float(derived_TKN(Ctraj, asm1)[0])
    sno = float(C[asm1.species_index["SNO"]])
    expected = 1.0e4 * (2.0 * tss + 1.0 * cod + 2.0 * bod
                        + 30.0 * tkn + 10.0 * sno) / 1000.0
    got = effluent_quality_index(t, Ctraj, Q, asm1)
    assert got == pytest.approx(expected, rel=1e-9)
