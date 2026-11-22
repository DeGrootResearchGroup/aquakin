"""Topological sort of plant units, plant.check() wiring validation, and
editing a plant's structure after a solve."""

import random

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import Aeration, Plant, PlantCheck
from aquakin.plant.bsm import build_bsm1
from aquakin.plant.bsm.bsm1 import BSM1_Q_AVG
from aquakin.plant.cstr import CSTRUnit
from aquakin.plant.mixer import MixerUnit, SetpointSplitter

_INF = {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
        "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


def _is_valid_eval_order(plant) -> bool:
    """Every non-recycle edge's source precedes its destination."""
    plant._finalize_topology()
    recycles = set(plant._recycle_keys)
    posn = {u: i for i, u in enumerate(plant._unit_order)}
    for c in plant.connections:
        if c.from_unit is None or (c.from_unit, c.from_port) in recycles:
            continue
        if posn[c.from_unit] >= posn[c.to_unit]:
            return False
    return True


# ----- topological sort (construction only; fast) -------------------------

def test_insertion_order_reproduced_when_already_topological(asm1):
    """A plant whose units were added in a valid order keeps that order and the
    same recycle set (backward-compatible)."""
    plant = build_bsm1(asm1)
    plant._finalize_topology()
    assert plant._unit_order == [
        "inlet_mix", "tank1", "tank2", "tank3", "tank4", "tank5",
        "tank5_split", "clarifier", "underflow_split"]
    assert set(plant._recycle_keys) == {
        ("tank5_split", "internal_recycle"), ("underflow_split", "ras")}


@pytest.mark.parametrize("seed", range(5))
def test_arbitrary_add_order_gives_valid_topology(asm1, seed):
    """Scrambling the add order still yields a valid evaluation order with the
    recycles as back-edges."""
    plant = build_bsm1(asm1)
    random.Random(seed).shuffle(plant._insertion_order)
    assert _is_valid_eval_order(plant)
    # Every back-edge is a real connection.
    conns = {(c.from_unit, c.from_port) for c in plant.connections}
    assert all(k in conns for k in plant._recycle_keys)


def test_recycle_built_in_reverse_order_solves(asm1):
    """A mix -> tank -> split loop with a split->mix recycle, built with the
    downstream units added *before* the upstream mixer (reverse of the old
    required order), produces a valid evaluation order and integrates -- the
    topological sort handles the cycle regardless of add order."""
    plant = Plant("rev", time_unit="d")
    # Add the downstream splitter first, then the tank, then the upstream mixer.
    plant.add_unit(SetpointSplitter(name="split", model=asm1,
                                    output_port_flows={"recycle": 1000.0},
                                    remainder_port="out"))
    plant.add_unit(CSTRUnit(name="tank", model=asm1, volume=1000.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}))
    plant.add_unit(MixerUnit(name="mix", input_port_names=["fresh", "recycle"],
                             model=asm1))
    plant.connect("mix", "tank")
    plant.connect("tank", "split")
    plant.connect("split.recycle", "mix.recycle")   # a cycle (any add order)
    plant.add_influent("feed", asm1.influent({"SS": 60.0}, Q=18446.0),
                       to="mix.fresh")
    plant._finalize_topology()
    # A valid evaluation order with exactly one back-edge breaking the cycle.
    assert _is_valid_eval_order(plant)
    assert len(plant._recycle_keys) == 1
    # And it integrates to a finite state.
    sol = plant.solve(t_span=(0.0, 5.0), t_eval=jnp.array([0.0, 5.0]),
                      rtol=1e-4, atol=1e-3)
    assert jnp.all(jnp.isfinite(sol.state))


# ----- plant.check() -------------------------------------------------------

def test_check_ok_on_built_plant(asm1):
    plant = build_bsm1(asm1)
    plant.add_influent("feed", asm1.influent({"SS": 60.0}, Q=18446.0))
    chk = plant.check()
    assert isinstance(chk, PlantCheck)
    assert chk.ok and chk.unfed_ports == []
    # The effluent + wastage leave the plant: reported as info, not errors.
    assert "clarifier.overflow" in chk.dangling_outputs
    assert set(chk.recycles) == {"tank5_split.internal_recycle",
                                 "underflow_split.ras"}


def test_check_flags_unfed_input_port(asm1):
    plant = Plant("broken", time_unit="d")
    plant.add_unit(CSTRUnit(name="t1", model=asm1, volume=1000.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}))
    plant.add_unit(CSTRUnit(name="t2", model=asm1, volume=1000.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}))
    plant.connect("t1", "t2")          # t1.inlet is never fed
    chk = plant.check()
    assert not chk.ok
    assert "t1.inlet" in chk.unfed_ports
    assert "t2.out" in chk.dangling_outputs
    with pytest.raises(ValueError, match="unfed input ports"):
        plant.check(raise_on_error=True)


# ----- functional: arbitrary add order solves identically (slow) ----------

@pytest.mark.slow
def test_scrambled_add_order_same_steady_state(asm1):
    def steady_snh(seed):
        plant = build_bsm1(asm1)
        if seed is not None:
            random.Random(seed).shuffle(plant._insertion_order)
        plant.add_influent("feed", asm1.influent(_INF, Q=BSM1_Q_AVG))
        ss = plant.run_to_steady_state(max_time=200.0)
        return float(plant.stream(ss.solution, plant.effluent_endpoint)
                     .C_named("SNH")[-1])

    ref = steady_snh(None)
    for seed in range(3):
        assert steady_snh(seed) == pytest.approx(ref, abs=1e-3)


# ----- editing a plant after a solve (fast) --------------------------------

def _add_second_tank(plant, model, **kwargs):
    """Wire a second CSTR after the plant's ``tank``."""
    plant.add_unit(CSTRUnit(name="tank2", model=model, volume=50.0,
                            input_port_names=["inlet"], conditions={"T": 293.15}, **kwargs))
    plant.connect("tank", "tank2")


def test_units_added_after_a_solve_match_a_fresh_plant(make_cstr_plant, simple_model):
    """A unit and a connection added after a solve are used by the next solve
    and by ``plant.stream``, exactly as on a plant built with them from the
    start (issue #749: the stream read raised ``KeyError: 'tank2'`` from the
    per-unit parameter cache of the first structure)."""
    t_eval = jnp.linspace(0.0, 20.0, 5)
    edited = make_cstr_plant()
    edited.solve(t_span=(0.0, 20.0), t_eval=t_eval)
    _add_second_tank(edited, simple_model)
    sol = edited.solve(t_span=(0.0, 20.0), t_eval=t_eval)

    fresh = make_cstr_plant()
    _add_second_tank(fresh, simple_model)
    ref = fresh.solve(t_span=(0.0, 20.0), t_eval=t_eval)

    np.testing.assert_allclose(np.asarray(sol.state), np.asarray(ref.state), rtol=1e-12)
    out = edited.stream(sol, "tank2.out")
    # A CSTR's outlet is its well-mixed state, at the flow it receives.
    np.testing.assert_allclose(np.asarray(out.C), np.asarray(sol.unit_state("tank2")),
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out.Q), 10.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out.C),
                               np.asarray(fresh.stream(ref, "tank2.out").C), rtol=1e-12)


def test_closed_loop_tank_added_after_a_solve_gets_its_controller(asm1):
    """A closed-loop aerated tank added after a solve gets the DO controller the
    plant derives for it, as it would before the first solve (the one-time
    controller set-up had already run, so the solve refused the tank's
    unpublished kLa signal)."""
    def first(plant):
        plant.add_unit(CSTRUnit(name="tank", model=asm1, volume=1000.0,
                                input_port_names=["inlet"], conditions={"T": 293.15}))
        plant.add_influent("feed", asm1.influent(_INF, Q=1000.0), to="tank.inlet")

    t_eval = jnp.linspace(0.0, 0.1, 3)
    edited = Plant("edited", time_unit="d")
    first(edited)
    edited.solve(t_span=(0.0, 0.1), t_eval=t_eval)
    _add_second_tank(edited, asm1, aeration=Aeration(do_setpoint=2.0))
    sol = edited.solve(t_span=(0.0, 0.1), t_eval=t_eval)

    fresh = Plant("fresh", time_unit="d")
    first(fresh)
    _add_second_tank(fresh, asm1, aeration=Aeration(do_setpoint=2.0))
    ref = fresh.solve(t_span=(0.0, 0.1), t_eval=t_eval)
    assert "tank2_aeration" in edited.list_units()
    assert edited.list_units() == fresh.list_units()
    np.testing.assert_allclose(np.asarray(sol.state), np.asarray(ref.state), rtol=1e-12)
