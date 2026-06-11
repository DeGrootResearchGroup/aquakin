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
    for name, val in (("eqi", ev.eqi), ("oci", ev.oci),
                      ("aeration_energy", ev.aeration_energy),
                      ("pumping_energy", ev.pumping_energy),
                      ("sludge_production", ev.sludge_production)):
        assert jnp.isfinite(val), f"{name} not finite: {val}"
        assert val > 0.0, f"{name} not positive: {val}"


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


def test_oci_is_sum_of_terms(evaluated):
    """OCI = aeration + pumping + 5 x sludge production (the BSM1-form index)."""
    _, _, _, ev = evaluated
    assert ev.oci == pytest.approx(
        ev.aeration_energy + ev.pumping_energy + 5.0 * ev.sludge_production,
        rel=1e-9)


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
