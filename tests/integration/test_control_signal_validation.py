"""Assembly-time validation of the control-signal bus.

A unit under closed-loop control reads ``signals[name]`` in its ``rhs``. If no
controller publishes ``name`` (a forgotten or mistyped wiring) that read is a
bare ``KeyError`` from deep inside the first jitted solve. The plant cross-checks
consumed-vs-published signal names at topology setup and raises a clear error
instead -- this pins that behaviour.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.control import PIController
from aquakin.plant.cstr import CSTRUnit
from aquakin.plant.plant import Plant


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_network("asm1")


def _controlled_tank(asm1, signal_name):
    return CSTRUnit(name="tank", network=asm1, volume=1000.0,
                    input_port_names=["in"], conditions={"T": 293.15},
                    kla={"SO": 240.0},
                    controlled_kla={"SO": (signal_name, 1.0)})


def _do_controller(asm1, signal_name):
    return PIController(
        name="do_ctrl", network=asm1, measured_species="SO", setpoint=2.0,
        Kp=25.0, Ti=0.002, Tt=0.001, offset=120.0, out_min=0.0, out_max=360.0,
        signal_name=signal_name)


def test_required_and_published_signal_names():
    """The declarative hooks the validation reads."""
    asm1 = aquakin.load_network("asm1")
    assert _controlled_tank(asm1, "do_signal").required_signals == ("do_signal",)
    assert _do_controller(asm1, "do_signal").signal_names == ("do_signal",)
    # An uncontrolled tank requires nothing.
    plain = CSTRUnit(name="t", network=asm1, volume=1.0, input_port_names=["in"],
                     conditions={"T": 293.15})
    assert plain.required_signals == ()


def test_matched_signal_passes(asm1):
    """A controlled tank with a controller publishing its signal validates."""
    p = Plant(name="ok")
    p.add_unit(_controlled_tank(asm1, "do_signal"))
    p.add_unit(_do_controller(asm1, "do_signal"))
    p._build_state_layout()        # runs _validate_control_signals; must not raise


def test_missing_publisher_raises_clearly(asm1):
    """A controlled tank whose signal nobody publishes fails at setup, naming the
    unit, the missing signal, and the (empty) published set -- not a deep KeyError."""
    p = Plant(name="bad")
    p.add_unit(_controlled_tank(asm1, "do_signal"))
    with pytest.raises(ValueError, match="consumes control signal 'do_signal'"):
        p._build_state_layout()


def test_typo_in_signal_name_raises(asm1):
    """A mismatch between the consumed and published name is caught."""
    p = Plant(name="typo")
    p.add_unit(_controlled_tank(asm1, "do_signl"))      # typo
    p.add_unit(_do_controller(asm1, "do_signal"))       # publishes the correct name
    with pytest.raises(ValueError, match="do_signl"):
        p._build_state_layout()


def test_unknown_publisher_skips_validation(asm1):
    """If a signal *producer* doesn't declare signal_names, the published set is
    unknown, so validation is skipped rather than risk rejecting a valid plant."""
    class CustomController:
        # A producer with signal_outputs but no signal_names declaration.
        name = "custom"
        state_size = 0
        input_ports: list = []
        output_ports: list = []

        def initial_state(self):
            return jnp.zeros((0,))

        def rhs(self, t, state, inputs, params, signals=None):
            return jnp.zeros((0,))

        def flow_outputs(self, input_flows, params, ctx=None):
            return {}

        def signal_outputs(self, t, state, inputs, params):
            return {"do_signal": jnp.asarray(1.0)}

    p = Plant(name="custom")
    p.add_unit(_controlled_tank(asm1, "do_signal"))
    p.add_unit(CustomController())
    p._build_state_layout()        # must not raise (publisher set is unknown)
