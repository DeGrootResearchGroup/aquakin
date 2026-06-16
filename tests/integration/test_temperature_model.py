"""Selectable plant TemperatureModel: algebraic (default) vs heat-balance state.

The heat-balance model gives each finite-volume unit a temperature state with the
completely-mixed first-order balance ``V dT/dt = Q_in (T_in - T)`` (the BSM2
protocol treatment), where the algebraic default takes the reactor temperature to
be the instantaneous flow-weighted inlet temperature. These tests pin:

- the default carries no extra state (byte-compatible with the historic plant);
- the heat-balance model tracks the right units and grows the state vector;
- the heat-balance derivative IS the first-order balance, with the right time
  constant ``V/Q`` (the lag is real and correctly sized);
- a constant-influent-temperature plant has the influent temperature as its
  heat-balance fixed point (so it reproduces the algebraic steady temperature);
- ``jax.grad`` flows through the temperature state.

All checks use single RHS evaluations (``plant.derivative``) -- no stiff solve --
so they run in the fast gate.
"""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.bsm.bsm2 import (
    bsm2_asm1_network,
    bsm2_constant_influent,
    bsm2_parameters,
    build_bsm2,
)
from aquakin.plant.cstr import Aeration, CSTRUnit
from aquakin.plant.plant import _TEMPERATURE_KEY


def _one_cstr_plant(asm1, *, volume, model, T_in_K, Q=1000.0):
    """A minimal single-CSTR plant fed one constant influent (output dangling)."""
    plant = aquakin.Plant("one_cstr")
    tank = CSTRUnit(name="tank", network=asm1, volume=volume,
                    input_port_names=["inlet"], conditions={"T": 288.15},
                    aeration=Aeration(kla=0.0))
    plant.add_unit(tank)
    infl = asm1.influent({}, Q=Q)
    infl = dataclasses.replace(infl, T=jnp.full_like(infl.Q, T_in_K))
    port = tank.input_ports[0]
    plant.add_influent("feed", infl, to=f"tank.{port}")
    plant.set_temperature_model(model)
    return plant, tank


# ----- default (algebraic) -------------------------------------------------

def test_algebraic_default_carries_no_temperature_state():
    """The default model adds no state -- the plant is unchanged."""
    plant = build_bsm2()
    plant._build_state_layout()
    assert plant._temperature_units == []
    assert plant._temperature_block[1] == 0
    assert _TEMPERATURE_KEY not in plant._split_state(plant.initial_state())


# ----- heat-balance tracked set + state growth -----------------------------

def test_heatbalance_tracks_volume_units_except_digester():
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    base = build_bsm2(asm1_network=asm1, adm1_network=adm1)
    base._build_state_layout()
    n_base = base._total_state_size

    hb = build_bsm2(asm1_network=asm1, adm1_network=adm1,
                    temperature_model=aquakin.HeatBalanceTemperature())
    hb._build_state_layout()
    # Every finite-volume liquid unit (5 reactors + primary + settler), NOT the
    # heated digester.
    assert set(hb._temperature_units) == {
        "primary", "tank1", "tank2", "tank3", "tank4", "tank5", "settler"}
    assert "digester" not in hb._temperature_units
    # The state grew by exactly one slot per tracked unit (tail append).
    assert hb._total_state_size == n_base + len(hb._temperature_units)


# ----- the heat balance + its time constant --------------------------------

def test_heatbalance_derivative_is_first_order_balance():
    """dT/dt for a single tracked CSTR equals (Q/V)(T_in - T)."""
    asm1 = bsm2_asm1_network()
    V, Q, T_in = 1500.0, 1000.0, 283.15      # 10 C influent
    plant, tank = _one_cstr_plant(
        asm1, volume=V, model=aquakin.HeatBalanceTemperature(), T_in_K=T_in, Q=Q)
    plant._build_state_layout()
    assert plant._temperature_units == ["tank"]
    y0 = plant.initial_state()               # tank temp seeded at its 15 C condition
    ts, tn = plant._temperature_block
    T_state = float(y0[ts])
    d = plant.derivative(y0, params=plant.default_parameters())
    dT = float(d[ts])
    assert dT == pytest.approx((Q / V) * (T_in - T_state), rel=1e-6)
    # Time constant tau = V/Q: the lag is real and correctly sized.
    tau = V / Q
    assert dT == pytest.approx((T_in - T_state) / tau, rel=1e-6)


def test_constant_influent_temperature_is_the_fixed_point():
    """At the influent temperature, every tracked unit's dT/dt is ~0 -- so the
    heat-balance steady temperature is the influent T, reproducing the algebraic
    instantaneous value (the no-regression-at-steady-state guarantee)."""
    asm1 = bsm2_asm1_network()
    adm1 = aquakin.load_network("adm1")
    T_in = 287.0
    # carbon=None so the only influent is the feed -> a single uniform inlet
    # temperature, whose fixed point is exactly that temperature (the external
    # carbon dose is a second influent at a different T, a tiny separate offset).
    plant = build_bsm2(asm1_network=asm1, adm1_network=adm1, carbon=None,
                       temperature_model=aquakin.HeatBalanceTemperature())
    infl = bsm2_constant_influent(asm1)
    infl = dataclasses.replace(infl, T=jnp.full_like(infl.Q, T_in))
    plant.add_influent("feed", infl, to=plant.influent_endpoint)
    plant._build_state_layout()
    from aquakin.plant.bsm import bsm2_warm_start
    y0 = bsm2_warm_start(plant)
    ts, tn = plant._temperature_block
    # Put every tracked temperature AT the influent temperature.
    y0 = y0.at[ts:ts + tn].set(T_in)
    d = plant.derivative(y0, params=bsm2_parameters(asm1, adm1))
    dT = d[ts:ts + tn]
    assert jnp.max(jnp.abs(dT)) < 1e-6      # influent T is the fixed point


# ----- AD ------------------------------------------------------------------

def test_grad_flows_through_temperature_state():
    """jax.grad of a reactor output w.r.t. the influent temperature flows through
    the heat-balance state without NaNs."""
    asm1 = bsm2_asm1_network()
    V, Q = 1500.0, 1000.0
    plant, tank = _one_cstr_plant(
        asm1, volume=V, model=aquakin.HeatBalanceTemperature(), T_in_K=283.15, Q=Q)
    plant._build_state_layout()
    params = plant.default_parameters()
    y0 = plant.initial_state()
    ts, _ = plant._temperature_block

    def loss(T_state):
        y = y0.at[ts].set(T_state)
        d = plant.derivative(y, params=params)
        return d[ts]                          # dT/dt = (Q/V)(T_in - T_state)

    g = jax.grad(loss)(float(y0[ts]))
    assert jnp.isfinite(g)
    assert float(g) == pytest.approx(-Q / V, rel=1e-6)   # d/dT_state of (Q/V)(T_in-T)
