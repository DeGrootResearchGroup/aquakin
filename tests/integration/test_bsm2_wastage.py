"""BSM2 scheduled (timed) wastage tests.

Covers the reusable :class:`PiecewiseConstantSchedule`, the time-threaded flow
solve, and the wired ``build_bsm2(wastage_schedule=...)`` plant whose
secondary-clarifier underflow steps the waste pump between a low and a high rate
on a schedule.
"""

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import (
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
    bsm2_wastage_schedule,
)
from aquakin.plant.bsm.bsm2 import BSM2_RAS, BSM2_WASTAGE_HIGH, BSM2_WASTAGE_LOW
from aquakin.plant.schedule import PiecewiseConstantSchedule


# ----- PiecewiseConstantSchedule (pure) -----------------------------------

def test_schedule_steps_between_values():
    s = PiecewiseConstantSchedule([182.0, 364.0], [300.0, 450.0, 300.0])
    assert float(s.at(0.0)) == 300.0
    assert float(s.at(181.9)) == 300.0
    assert float(s.at(182.0)) == 450.0      # steps at the break (side='right')
    assert float(s.at(363.9)) == 450.0
    assert float(s.at(364.0)) == 300.0
    assert float(s.at(1000.0)) == 300.0


def test_schedule_length_validation():
    with pytest.raises(ValueError, match="len.values."):
        PiecewiseConstantSchedule([1.0, 2.0], [10.0, 20.0])  # need 3 values


def test_schedule_requires_increasing_breaks():
    with pytest.raises(ValueError, match="strictly increasing"):
        PiecewiseConstantSchedule([2.0, 1.0], [1.0, 2.0, 3.0])


def test_schedule_shifted_offsets_values():
    s = PiecewiseConstantSchedule([182.0], [300.0, 450.0]).shifted(20648.0)
    assert float(s.at(0.0)) == pytest.approx(20948.0)
    assert float(s.at(200.0)) == pytest.approx(21098.0)


def test_schedule_is_jittable():
    s = PiecewiseConstantSchedule([182.0, 364.0], [300.0, 450.0, 300.0])
    f = jax.jit(lambda t: s.at(t))
    assert float(f(200.0)) == 450.0


def test_bsm2_wastage_schedule_values():
    s = bsm2_wastage_schedule()
    assert float(s.at(100.0)) == BSM2_WASTAGE_LOW
    assert float(s.at(250.0)) == BSM2_WASTAGE_HIGH
    assert float(s.at(450.0)) == BSM2_WASTAGE_LOW
    assert float(s.at(600.0)) == BSM2_WASTAGE_HIGH


# ----- Wired BSM2 plant with the scheduled wastage ------------------------

WARM_AS = {"SI": 28.06, "SS": 2.0, "XI": 1532.3, "XS": 45.0, "XB_H": 2244.0,
           "XB_A": 167.0, "XP": 967.0, "SO": 1.0, "SNO": 7.0, "SNH": 3.0,
           "SND": 0.7, "XND": 3.0, "SALK": 5.0}


@pytest.fixture(scope="module")
def asm1():
    return bsm2_asm1_network()


@pytest.fixture(scope="module")
def adm1():
    return aquakin.load_network("adm1")


@pytest.fixture(scope="module")
def wastage_run(asm1, adm1):
    """Integrate across the first schedule step (day 182), sampling either side."""
    plant = build_bsm2(asm1, adm1, wastage_schedule=bsm2_wastage_schedule())
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")
    params = bsm2_parameters(asm1, adm1)
    warm = asm1.concentrations(WARM_AS)
    tanks = ("tank1", "tank2", "tank3", "tank4", "tank5")
    y0 = plant.initial_state(overrides={t: warm for t in tanks})
    sol = plant.solve((0.0, 250.0), t_eval=jnp.array([0.0, 150.0, 250.0]),
                      params=params, y0=jnp.asarray(y0),
                      rtol=1e-5, atol=1e-3, max_steps=800_000)
    return plant, sol, params


def test_wastage_run_finite(wastage_run):
    _, sol, _ = wastage_run
    assert jnp.all(jnp.isfinite(sol.state))


def test_waste_flow_steps_on_schedule(wastage_run):
    """The waste pump is Qw_low before day 182 and Qw_high after; RAS stays
    fixed (the schedule moves only the wastage, not the recycle)."""
    plant, sol, params = wastage_run
    waste = plant.stream(sol, "underflow_split.waste", params)
    ras = plant.stream(sol, "underflow_split.ras", params)
    # Sampled at days [0, 150, 250].
    assert float(waste.Q[1]) == pytest.approx(BSM2_WASTAGE_LOW, abs=1e-3)   # day 150
    assert float(waste.Q[2]) == pytest.approx(BSM2_WASTAGE_HIGH, abs=1e-3)  # day 250
    assert float(ras.Q[1]) == pytest.approx(BSM2_RAS, rel=1e-6)
    assert float(ras.Q[2]) == pytest.approx(BSM2_RAS, rel=1e-6)


def test_higher_wastage_lowers_biomass(wastage_run):
    """Stepping the wastage up wastes more sludge, so the reactor biomass falls
    relative to the constant-Qw plant (the point of the wastage schedule)."""
    plant, sol, params = wastage_run
    asm1 = plant.units["tank1"].network
    adm1 = aquakin.load_network("adm1")
    # Constant-Qw reference over the same window and warm start.
    ref = build_bsm2(asm1, adm1)
    ref.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")
    warm = asm1.concentrations(WARM_AS)
    tanks = ("tank1", "tank2", "tank3", "tank4", "tank5")
    y0 = ref.initial_state(overrides={t: warm for t in tanks})
    sol_ref = ref.solve((0.0, 250.0), t_eval=jnp.array([0.0, 250.0]),
                        params=params, y0=jnp.asarray(y0),
                        rtol=1e-5, atol=1e-3, max_steps=800_000)
    xbh_sched = float(sol.C_named("tank5", "XB_H")[-1])
    xbh_const = float(sol_ref.C_named("tank5", "XB_H")[-1])
    assert xbh_sched < xbh_const
