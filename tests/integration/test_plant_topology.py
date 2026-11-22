"""Topological sort of plant units + plant.check() wiring validation."""

import random

import jax.numpy as jnp
import numpy as np
import pytest

import aquakin
from aquakin.plant import Plant, PlantCheck
from aquakin.plant.bsm import bsm1_warm_start, build_bsm1
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


def _mix_tank_split_loop(asm1, *, reverse):
    """A mix -> tank -> split loop with a split -> mix recycle. With ``reverse``
    the downstream splitter is added first, then the tank, then the upstream
    mixer (the reverse of the order the cycle is wired in)."""
    plant = Plant("rev" if reverse else "fwd", time_unit="d")
    units = [
        MixerUnit(name="mix", input_port_names=["fresh", "recycle"], model=asm1),
        CSTRUnit(name="tank", model=asm1, volume=1000.0,
                 input_port_names=["inlet"], conditions={"T": 293.15}),
        SetpointSplitter(name="split", model=asm1,
                         output_port_flows={"recycle": 1000.0}, remainder_port="out"),
    ]
    for unit in (units[::-1] if reverse else units):
        plant.add_unit(unit)
    plant.connect("mix", "tank")
    plant.connect("tank", "split")
    plant.connect("split.recycle", "mix.recycle")   # a cycle (any add order)
    plant.add_influent("feed", asm1.influent({"SS": 60.0}, Q=18446.0),
                       to="mix.fresh")
    return plant


def test_recycle_built_in_reverse_order_solves(asm1):
    """The loop built in reverse add order gets a valid evaluation order with
    exactly one back-edge, and is the same plant as the one built in forward
    order: the same right-hand side at an arbitrary state (to the recycle
    solve's 1e-8 tolerance; the two cut the cycle at different edges) and the
    same 5-day trajectory (a decade above the solve's ``rtol``)."""
    plant = _mix_tank_split_loop(asm1, reverse=True)
    plant._finalize_topology()
    assert _is_valid_eval_order(plant)
    assert len(plant._recycle_keys) == 1
    ref = _mix_tank_split_loop(asm1, reverse=False)

    rng = np.random.default_rng(0)
    y = ref.initial_state() * jnp.asarray(rng.uniform(0.5, 1.5, ref.initial_state().shape)) + 1.0
    d_ref = np.asarray(ref.derivative(y))
    assert np.max(np.abs(d_ref)) > 0.0
    np.testing.assert_allclose(np.asarray(plant.derivative(y)), d_ref, rtol=1e-8)

    kw = dict(t_span=(0.0, 5.0), t_eval=jnp.array([0.0, 5.0]), rtol=1e-4, atol=1e-3)
    sol = plant.solve(**kw)
    sol_ref = ref.solve(**kw)
    np.testing.assert_allclose(np.asarray(sol.final_state), np.asarray(sol_ref.final_state),
                               rtol=1e-3)


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


# ----- functional: arbitrary add order, same steady state (slow) ----------

@pytest.mark.slow
def test_scrambled_add_order_same_steady_state(asm1):
    """BSM1 with its units added in a scrambled order has the same dynamics as
    the plant built in order: at the reference plant's steady state (one PTC
    solve) every shuffled plant's right-hand side is zero to the solve's
    tolerance, and at the warm start (far from steady) it equals the reference
    right-hand side unit by unit. A topology changed by the add order would move
    both."""
    def bsm1(seed):
        plant = build_bsm1(asm1)
        if seed is not None:
            random.Random(seed).shuffle(plant._insertion_order)
        plant.add_influent("feed", asm1.influent(_INF, Q=BSM1_Q_AVG))
        return plant

    ref = bsm1(None)
    tol = 1e-10
    ss = ref.steady_state(y0=bsm1_warm_start(ref), tol=tol)
    assert ss.method == "ptc" and bool(ss.converged)
    ss_units = ref.states_by_unit(ss.state)
    warm_units = ref.states_by_unit(bsm1_warm_start(ref))
    d_warm_ref = ref.states_by_unit(ref.derivative(bsm1_warm_start(ref)))
    for seed in range(3):
        plant = bsm1(seed)
        y_ss = plant.initial_state(overrides=ss_units)
        # The scaled residual steady_state converges on, with its default floor.
        scaled = jnp.abs(plant.derivative(y_ss)) / jnp.maximum(jnp.abs(y_ss), 1e-6)
        assert float(jnp.max(scaled)) <= 10 * tol, seed
        d_warm = plant.states_by_unit(plant.derivative(plant.initial_state(overrides=warm_units)))
        for name, d in d_warm_ref.items():
            np.testing.assert_allclose(np.asarray(d_warm[name]), np.asarray(d),
                                       rtol=1e-8, atol=1e-8, err_msg=f"{seed} {name}")
