"""BSM2 reject storage-tank tests.

Two layers: the :class:`StorageTank` level-gated bypass logic (fast, no plant
solve) and the wired ``build_bsm2(reject_storage=True)`` plant (one
module-scoped solve, since the suite runs near the CI runner's limit). The key
faithfulness property: with the default zero release the open-loop tank fills to
its upper limit and bypasses all reject, so the steady state is unchanged.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import (
    build_bsm2,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
)
from aquakin.plant.storage import StorageTank
from aquakin.plant.streams import Stream


@pytest.fixture(scope="module")
def asm1():
    return bsm2_asm1_network()


def _tank(asm1, output_flow=0.0, volume=160.0):
    return StorageTank(name="store", network=asm1, volume=volume,
                       output_flow=output_flow)


def _state(tank, asm1, V):
    """A storage state at liquid volume ``V`` (default concentrations)."""
    return jnp.concatenate([asm1.default_concentrations(), jnp.asarray([V])])


def _stream(asm1, Q):
    return {"in": Stream(
        Q=jnp.asarray(float(Q)), C=asm1.default_concentrations(), network=asm1)}


# ----- StorageTank regimes (no plant solve) -------------------------------

def test_state_size_and_initial_volume(asm1):
    tank = _tank(asm1, volume=160.0)
    assert tank.state_size == asm1.n_species + 1
    y0 = tank.initial_state()
    assert float(y0[-1]) == pytest.approx(0.5 * 160.0)  # initial_fraction default


def test_normal_regime_releases_requested_flow(asm1):
    """Mid-level: release the requested flow, no bypass, tank accumulates."""
    tank = _tank(asm1, output_flow=200.0)
    state = _state(tank, asm1, V=80.0)        # mid (0.1*160 < 80 < 0.9*160)
    outs = tank.compute_outputs(0.0, state, _stream(asm1, 500.0), None)
    assert float(outs["out"].Q) == pytest.approx(200.0)
    assert float(outs["bypass"].Q) == pytest.approx(0.0)
    dV = float(tank.rhs(0.0, state, _stream(asm1, 500.0), None)[-1])
    assert dV == pytest.approx(500.0 - 200.0)  # filling


def test_full_and_filling_bypasses(asm1):
    """At the upper limit with inflow > release, divert everything; volume held."""
    tank = _tank(asm1, output_flow=0.0)
    state = _state(tank, asm1, V=0.9 * 160.0)
    s = _stream(asm1, 438.0)
    outs = tank.compute_outputs(0.0, state, s, None)
    assert float(outs["bypass"].Q) == pytest.approx(438.0)
    assert float(outs["out"].Q) == pytest.approx(0.0)
    assert float(tank.rhs(0.0, state, s, None)[-1]) == pytest.approx(0.0)


def test_full_and_draining_is_normal(asm1):
    """At the upper limit but draining (release > inflow), release normally."""
    tank = _tank(asm1, output_flow=600.0)
    state = _state(tank, asm1, V=0.9 * 160.0)
    s = _stream(asm1, 400.0)
    outs = tank.compute_outputs(0.0, state, s, None)
    assert float(outs["out"].Q) == pytest.approx(600.0)
    assert float(outs["bypass"].Q) == pytest.approx(0.0)
    assert float(tank.rhs(0.0, state, s, None)[-1]) == pytest.approx(400.0 - 600.0)


def test_empty_regime_stops_releasing(asm1):
    """At the lower limit: stop releasing and just fill."""
    tank = _tank(asm1, output_flow=600.0)
    state = _state(tank, asm1, V=0.1 * 160.0)
    s = _stream(asm1, 400.0)
    outs = tank.compute_outputs(0.0, state, s, None)
    assert float(outs["out"].Q) == pytest.approx(0.0)
    assert float(outs["bypass"].Q) == pytest.approx(0.0)
    assert float(tank.rhs(0.0, state, s, None)[-1]) == pytest.approx(400.0)  # fills


@pytest.mark.parametrize("V,Qreq,Qin", [
    (80.0, 200.0, 500.0), (144.0, 0.0, 438.0), (144.0, 600.0, 400.0),
    (16.0, 600.0, 400.0)])
def test_flow_volume_conservation(asm1, V, Qreq, Qin):
    """out + bypass + dV/dt == Q_in in every regime (nothing created/lost)."""
    tank = _tank(asm1, output_flow=Qreq)
    state = _state(tank, asm1, V)
    s = _stream(asm1, Qin)
    outs = tank.compute_outputs(0.0, state, s, None)
    dV = float(tank.rhs(0.0, state, s, None)[-1])
    total = float(outs["out"].Q) + float(outs["bypass"].Q) + dV
    assert total == pytest.approx(Qin)


def test_flow_outputs_match_compute_outputs(asm1):
    """The flow-network rule agrees with the concentration-stage flows."""
    tank = _tank(asm1, output_flow=200.0)
    state = _state(tank, asm1, V=80.0)
    flows = tank.flow_outputs({"in": jnp.asarray(500.0)}, None, state)
    outs = tank.compute_outputs(0.0, state, _stream(asm1, 500.0), None)
    assert float(flows["out"]) == pytest.approx(float(outs["out"].Q))
    assert float(flows["bypass"]) == pytest.approx(float(outs["bypass"].Q))


# ----- Wired BSM2 plant with reject storage -------------------------------

WARM_AS = {"SI": 28.06, "SS": 2.0, "XI": 1532.3, "XS": 45.0, "XB_H": 2244.0,
           "XB_A": 167.0, "XP": 967.0, "SO": 1.0, "SNO": 7.0, "SNH": 3.0,
           "SND": 0.7, "XND": 3.0, "SALK": 5.0}


@pytest.fixture(scope="module")
def adm1():
    return aquakin.load_network("adm1")


@pytest.fixture(scope="module")
def storage_run(asm1, adm1):
    plant = build_bsm2(asm1, adm1, reject_storage=True)
    plant.add_influent("feed", bsm2_constant_influent(asm1), to="front_mix.fresh")
    params = bsm2_parameters(asm1, adm1)
    warm = asm1.concentrations(WARM_AS)
    tanks = ("tank1", "tank2", "tank3", "tank4", "tank5")
    y0 = plant.initial_state(overrides={t: warm for t in tanks})
    sol = plant.solve((0.0, 80.0), t_eval=jnp.array([0.0, 80.0]),
                      params=params, y0=jnp.asarray(y0),
                      rtol=1e-5, atol=1e-3, max_steps=500_000)
    return plant, sol, params


def test_storage_plant_is_finite_and_healthy(storage_run):
    plant, sol, _ = storage_run
    assert jnp.all(jnp.isfinite(sol.state))
    assert "reject_storage" in plant.units
    # Nitrifying activated sludge sustained through the storage-routed reject.
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 1000.0
    assert float(sol.C_named("tank5", "SNH")[-1]) < 5.0


def test_storage_fills_and_bypasses_all_reject(storage_run):
    """Open-loop (zero release): the tank fills to its upper limit and bypasses
    the whole reject stream, so it is a faithful pass-through at steady state."""
    plant, sol, params = storage_run
    V = float(sol.unit_state("reject_storage")[-1, -1])
    assert V == pytest.approx(0.9 * 160.0, abs=1.0)  # full level
    bypass = plant.stream(sol, "reject_storage.bypass", params)
    released = plant.stream(sol, "reject_storage.out", params)
    reject_in = plant.stream(sol, "reject_mix.out", params)
    assert float(released.Q[-1]) == pytest.approx(0.0, abs=1e-6)
    assert float(bypass.Q[-1]) == pytest.approx(float(reject_in.Q[-1]), rel=1e-6)
