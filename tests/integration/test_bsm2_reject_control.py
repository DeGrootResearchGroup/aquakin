"""BSM2 closed-loop reject control (storage-tank level controller).

The reject storage tank can run a proportional level controller on its release:
it holds a mid-level setpoint and releases the reject smoothly, instead of
filling to its limit and bypassing. Two layers: the level-control release law
(fast, no plant solve) and the wired ``build_bsm2(reject_control=True)`` plant
(one module-scoped solve).
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import (
    bsm2_warm_start,
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
)
from aquakin.plant.bsm.bsm2 import BSM2_STORAGE_VOLUME
from aquakin.plant.storage import StorageTank
from aquakin.plant.streams import Stream
from aquakin.plant.units import FlowContext


@pytest.fixture(scope="module")
def asm1():
    return bsm2_asm1_network()


def _controlled_tank(asm1, setpoint=80.0, gain=30.0, bias=0.0, qmax=1500.0):
    return StorageTank(name="store", network=asm1, volume=160.0,
                       level_setpoint=setpoint, level_gain=gain,
                       output_flow_bias=bias, output_flow_max=qmax)


def _state(asm1, V):
    return jnp.concatenate([asm1.default_concentrations(), jnp.asarray([V])])


def _stream(asm1, Q):
    return {"in": Stream(Q=jnp.asarray(float(Q)), C=asm1.default_concentrations(),
                         network=asm1)}


# ----- Level-control release law (no plant solve) -------------------------

def test_release_proportional_to_level(asm1):
    """Above the setpoint the release rises with the level (P control)."""
    tank = _controlled_tank(asm1, setpoint=80.0, gain=30.0)
    assert float(tank._release_request(jnp.asarray(80.0))) == pytest.approx(0.0)
    assert float(tank._release_request(jnp.asarray(90.0))) == pytest.approx(300.0)
    assert float(tank._release_request(jnp.asarray(100.0))) == pytest.approx(600.0)


def test_release_clamped_to_pump_capacity(asm1):
    """The release saturates at the pump capacity and never goes negative."""
    tank = _controlled_tank(asm1, setpoint=80.0, gain=30.0, qmax=1500.0)
    assert float(tank._release_request(jnp.asarray(200.0))) == pytest.approx(1500.0)
    assert float(tank._release_request(jnp.asarray(40.0))) == pytest.approx(0.0)


def test_controlled_release_appears_at_outlet(asm1):
    """Mid-level, inflow below the release: the outlet carries the level-law
    release (the tank drains), no bypass."""
    tank = _controlled_tank(asm1, setpoint=80.0, gain=30.0)
    state = _state(asm1, V=100.0)            # release request = 600
    outs = tank.compute_outputs(0.0, state, _stream(asm1, 438.0), None)
    assert float(outs["out"].Q) == pytest.approx(600.0)
    assert float(outs["bypass"].Q) == pytest.approx(0.0)


def test_controlled_flow_volume_conservation(asm1):
    """out + bypass + dV/dt == Q_in under level control too."""
    tank = _controlled_tank(asm1, setpoint=80.0, gain=30.0)
    state = _state(asm1, V=95.0)
    s = _stream(asm1, 438.0)
    outs = tank.compute_outputs(0.0, state, s, None)
    dV = float(tank.rhs(0.0, state, s, None)[-1])
    total = float(outs["out"].Q) + float(outs["bypass"].Q) + dV
    assert total == pytest.approx(438.0)


def test_flow_outputs_match_compute_outputs(asm1):
    tank = _controlled_tank(asm1, setpoint=80.0, gain=30.0)
    state = _state(asm1, V=100.0)
    flows = tank.flow_outputs({"in": jnp.asarray(438.0)}, None, FlowContext(state=state))
    outs = tank.compute_outputs(0.0, state, _stream(asm1, 438.0), None)
    assert float(flows["out"]) == pytest.approx(float(outs["out"].Q))
    assert float(flows["bypass"]) == pytest.approx(float(outs["bypass"].Q))


# ----- Wired BSM2 plant with closed-loop reject control -------------------


@pytest.fixture(scope="module")
def adm1():
    return aquakin.load_network("adm1")


@pytest.fixture(scope="module")
def control_run(asm1, adm1):
    plant = build_bsm2(asm1, adm1, reject_control=True)
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")
    params = bsm2_parameters(asm1, adm1)
    y0 = bsm2_warm_start(plant)
    sol = plant.solve((0.0, 80.0), t_eval=jnp.array([0.0, 80.0]),
                      params=params, y0=jnp.asarray(y0),
                      rtol=1e-5, atol=1e-3, max_steps=600_000)
    return plant, sol, params


def test_reject_control_plant_finite_and_healthy(control_run):
    plant, sol, _ = control_run
    assert jnp.all(jnp.isfinite(sol.state))
    assert "reject_storage" in plant.units
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 1000.0
    assert float(sol.C_named("tank5", "SNH")[-1]) < 5.0


def test_tank_holds_midlevel_and_releases_reject(control_run):
    """Unlike the open-loop tank (which fills and bypasses), the controlled tank
    holds a mid-level and releases the reject -- a functioning buffer."""
    plant, sol, params = control_run
    V = float(sol.unit_state("reject_storage")[-1, -1])
    # Held away from the upper (0.9*Vmax) overflow limit.
    assert 0.1 * BSM2_STORAGE_VOLUME < V < 0.9 * BSM2_STORAGE_VOLUME
    released = plant.stream(sol, "reject_storage.out", params)
    bypass = plant.stream(sol, "reject_storage.bypass", params)
    reject_in = plant.stream(sol, "reject_mix.out", params)
    # The reject leaves via the controlled release, not the overflow bypass.
    assert float(bypass.Q[-1]) == pytest.approx(0.0, abs=1.0)
    assert float(released.Q[-1]) == pytest.approx(float(reject_in.Q[-1]), rel=1e-3)
