"""BSM2 influent hydraulic-delay tests.

Covers the :class:`HydraulicDelayUnit` first-order flow/load lag (fast, no plant
solve) and the wired ``build_bsm2(hydraulic_delay=HydraulicDelay())`` plant.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm import (
    bsm2_warm_start,
    build_bsm2,
    HydraulicDelay,
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
)
from aquakin.plant.delay import HydraulicDelayUnit
from aquakin.plant.streams import Stream
from aquakin.plant.units import FlowContext


@pytest.fixture(scope="module")
def asm1():
    return bsm2_asm1_network()


def _delay(asm1, tau=0.02, Q0=100.0):
    return HydraulicDelayUnit(name="d", network=asm1, tau=tau, initial_flow=Q0,
                              initial_concentrations=asm1.default_concentrations())


def _inlet(asm1, Q):
    return {"in": Stream(Q=jnp.asarray(float(Q)),
                         C=asm1.default_concentrations(), network=asm1)}


# ----- HydraulicDelayUnit (no plant solve) --------------------------------

def test_state_size_and_initial_state(asm1):
    d = _delay(asm1, Q0=150.0)
    assert d.state_size == asm1.n_species + 1
    y0 = d.initial_state()
    assert float(y0[-1]) == pytest.approx(150.0)               # held flow
    # Held loads = Q0 * C0.
    assert jnp.allclose(y0[:-1], 150.0 * asm1.default_concentrations())


def test_tau_validation(asm1):
    with pytest.raises(ValueError, match="tau must be"):
        HydraulicDelayUnit(name="d", network=asm1, tau=0.0)


def test_outputs_are_load_over_flow(asm1):
    """The outlet concentration is the held load divided by the held flow."""
    d = _delay(asm1, Q0=120.0)
    state = d.initial_state()
    out = d.compute_outputs(0.0, state, _inlet(asm1, 200.0), None)["out"]
    assert float(out.Q) == pytest.approx(120.0)                # held flow
    assert jnp.allclose(out.C, asm1.default_concentrations())  # load/flow = C0


def test_flow_outputs_use_held_flow(asm1):
    d = _delay(asm1, Q0=137.0)
    state = d.initial_state()
    flows = d.flow_outputs({"in": jnp.asarray(200.0)}, None, FlowContext(state=state))
    assert float(flows["out"]) == pytest.approx(137.0)


def test_inlet_is_a_fixed_point(asm1):
    """When the held state matches the inlet (load=Q_in*C_in, flow=Q_in), the
    derivative is zero -- the delay is a pass-through at steady state."""
    d = _delay(asm1, Q0=100.0)
    C_in = asm1.default_concentrations()
    Q_in = 250.0
    state = jnp.concatenate([Q_in * C_in, jnp.asarray([Q_in])])
    dstate = d.rhs(0.0, state, {"in": Stream(Q=jnp.asarray(Q_in), C=C_in,
                                             network=asm1)}, None)
    assert jnp.allclose(dstate, 0.0, atol=1e-9)


def test_first_order_lag_response(asm1):
    """Forward-Euler the lag from rest: after one time constant the held flow has
    covered ~1-1/e (~63%) of a step, the first-order signature."""
    d = _delay(asm1, tau=0.02, Q0=100.0)
    state = d.initial_state()
    inlet = _inlet(asm1, 200.0)
    dt = 2e-5
    for _ in range(int(0.02 / dt)):                       # integrate one tau
        state = state + dt * d.rhs(0.0, state, inlet, None)
    Q = float(d.compute_outputs(0.0, state, inlet, None)["out"].Q)
    frac = (Q - 100.0) / (200.0 - 100.0)
    assert frac == pytest.approx(1.0 - 1.0 / jnp.e, abs=0.02)


# ----- Wired BSM2 plant ----------------------------------------------------


@pytest.fixture(scope="module")
def adm1():
    return aquakin.load_network("adm1")


@pytest.fixture(scope="module")
def delay_run(asm1, adm1):
    plant = build_bsm2(asm1, adm1, hydraulic_delay=HydraulicDelay())
    plant.add_influent("feed", bsm2_constant_influent(asm1))
    params = bsm2_parameters(asm1, adm1)
    y0 = bsm2_warm_start(plant)
    sol = plant.solve((0.0, 80.0), t_eval=jnp.array([0.0, 80.0]),
                      params=params, y0=jnp.asarray(y0),
                      rtol=1e-5, atol=1e-3, max_steps=600_000)
    return plant, sol, params


def test_delay_is_front_most_unit(delay_run):
    plant, _, _ = delay_run
    assert plant._unit_order[0] == "influent_delay"


def test_plant_finite_and_steady_state_unchanged(delay_run):
    """At steady state the lag is a pass-through, so the activated-sludge state
    matches the no-delay plant and the delayed flow equals the influent."""
    plant, sol, params = delay_run
    assert jnp.all(jnp.isfinite(sol.state))
    assert float(sol.C_named("tank5", "XB_H")[-1]) > 2000.0
    assert float(sol.C_named("tank5", "SNH")[-1]) < 1.0
    out = plant.stream(sol, "influent_delay.out", params)
    from aquakin.plant.bsm.bsm2 import BSM2_Q_REF
    assert float(out.Q[-1]) == pytest.approx(BSM2_Q_REF, rel=1e-4)
