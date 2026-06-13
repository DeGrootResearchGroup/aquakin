"""The Aeration abstraction on CSTRUnit (issue #137).

``Aeration`` replaces the raw per-species ``kla`` / ``C_sat`` dicts with the
quantity a designer thinks in -- a fixed mass-transfer coefficient (open loop) or
a dissolved-oxygen setpoint (closed loop), the latter auto-wiring a PI controller
on the plant. Covers the spec validation, the open-loop translation, and the
closed-loop auto-wiring (per-tank and shared-controller).
"""
import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant import Aeration
from aquakin.plant.cstr import CSTRUnit
from aquakin.plant.plant import Plant


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


def _tank(asm1, name, aeration, **kw):
    return CSTRUnit(name=name, network=asm1, volume=1000.0,
                    input_port_names=["in"], conditions={"T": 293.15},
                    aeration=aeration, **kw)


# --- spec validation --------------------------------------------------------

def test_requires_exactly_one_mode():
    with pytest.raises(ValueError, match="exactly one"):
        Aeration()                                   # neither
    with pytest.raises(ValueError, match="exactly one"):
        Aeration(kla=120.0, do_setpoint=2.0)         # both
    with pytest.raises(ValueError, match="kla must be"):
        Aeration(kla=-1.0)


# --- open loop --------------------------------------------------------------

def test_open_loop_sets_kla_and_saturation(asm1):
    tank = _tank(asm1, "t", Aeration(kla=240.0, do_sat=8.0))
    so = asm1.species_index["SO"]
    assert float(tank._kla_vec[so]) == 240.0
    assert float(tank._sat_vec[so]) == 8.0
    assert tank.required_signals == ()                # no control signal
    assert tank._controlled_kla == {}


def test_do_sat_defaults_to_eight(asm1):
    tank = _tank(asm1, "t", Aeration(kla=120.0))
    assert float(tank._sat_vec[asm1.species_index["SO"]]) == 8.0


def test_anoxic_tank_has_no_aeration(asm1):
    tank = _tank(asm1, "t", None)
    assert float(tank._kla_vec.max()) == 0.0
    assert float(tank._sat_vec.max()) == 0.0
    assert tank.required_signals == ()


# --- closed loop: per-tank auto-wiring --------------------------------------

def test_per_tank_do_setpoint_auto_wires_its_own_controller(asm1):
    p = Plant(name="pertank")
    p.add_unit(_tank(asm1, "reactor", Aeration(do_setpoint=2.0)))
    p._build_state_layout()                           # materialises + validates
    # a dedicated controller, named off the tank, sensing the tank itself
    assert "reactor_aeration" in p.units
    ctrl = p.units["reactor_aeration"]
    assert ctrl.setpoint == 2.0 and ctrl.measured_species == "SO"
    assert ctrl.signal_names == ("_aer_reactor_kla",)
    assert p.units["reactor"].required_signals == ("_aer_reactor_kla",)
    # the sensor tap was wired: reactor -> controller.measured
    assert any(c.from_unit == "reactor" and c.to_unit == "reactor_aeration"
               for c in p.connections)


def test_materialisation_is_idempotent(asm1):
    p = Plant(name="idem")
    p.add_unit(_tank(asm1, "reactor", Aeration(do_setpoint=2.0)))
    p._build_state_layout()
    p._build_state_layout()                           # second solve setup
    n_ctrl = sum(1 for u in p.units.values() if hasattr(u, "setpoint"))
    assert n_ctrl == 1                                # not added twice


# --- closed loop: shared controller -----------------------------------------

def test_shared_controller_drives_several_tanks(asm1):
    p = Plant(name="shared")
    p.add_unit(_tank(asm1, "tankA",
               Aeration(do_setpoint=2.0, controller="do", sensor="tankA", gain=1.0)))
    p.add_unit(_tank(asm1, "tankB",
               Aeration(do_setpoint=2.0, controller="do", sensor="tankA", gain=0.5)))
    p._build_state_layout()
    # exactly one controller, named after the shared id, sensing tankA
    ctrls = [n for n, u in p.units.items() if hasattr(u, "setpoint")]
    assert ctrls == ["do"]
    # both tanks read the same signal; gains differ
    assert p.units["tankA"]._controlled_kla["SO"] == ("_aer_do_kla", 1.0)
    assert p.units["tankB"]._controlled_kla["SO"] == ("_aer_do_kla", 0.5)


def test_shared_controller_disagreement_raises(asm1):
    p = Plant(name="bad")
    p.add_unit(_tank(asm1, "tankA",
               Aeration(do_setpoint=2.0, controller="do", sensor="tankA")))
    p.add_unit(_tank(asm1, "tankB",
               Aeration(do_setpoint=1.5, controller="do", sensor="tankA")))  # differs
    with pytest.raises(ValueError, match="must agree"):
        p._build_state_layout()


def test_sensor_must_exist(asm1):
    p = Plant(name="nosensor")
    p.add_unit(_tank(asm1, "tankA",
               Aeration(do_setpoint=2.0, controller="do", sensor="ghost")))
    with pytest.raises(ValueError, match="senses unit 'ghost'"):
        p._build_state_layout()
