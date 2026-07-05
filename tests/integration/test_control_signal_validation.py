"""Assembly-time wiring of the control-signal bus.

A unit under closed-loop control reads ``signals[name]`` in its ``rhs``. Two
things keep that safe at assembly time:

- A ``CSTRUnit`` with a closed-loop :class:`~aquakin.plant.Aeration` spec has its
  controller **auto-wired** by the plant (``_materialize_aeration``), so the
  signal it consumes is always published -- there is no forgotten-controller
  footgun for the common case.
- The generic cross-check still runs for any other consumer: a unit that
  declares ``required_signals`` no producer publishes fails at topology setup
  with a clear error, not a deep ``KeyError`` from the first jitted solve.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant import Aeration
from aquakin.plant.control import PIController
from aquakin.plant.cstr import CSTRUnit
from aquakin.plant.plant import Plant


@pytest.fixture(scope="module")
def asm1():
    return aquakin.load_model("asm1")


def _do_controller(asm1, signal_name):
    return PIController(
        name="do_ctrl", model=asm1, measured_species="SO", setpoint=2.0,
        Kp=25.0, Ti=0.002, Tt=0.001, offset=120.0, out_min=0.0, out_max=360.0,
        signal_name=signal_name)


class _SignalConsumer:
    """A minimal mock unit that consumes a control signal but is not a CSTRUnit,
    so the plant does not auto-wire a controller for it -- used to exercise the
    generic consumed-vs-published validation."""

    def __init__(self, signal_name):
        self.name = "consumer"
        self._signal = signal_name

    state_size = 0
    input_ports: list = []
    output_ports: list = []

    @property
    def required_signals(self):
        return (self._signal,)

    def initial_state(self):
        return jnp.zeros((0,))

    def compute_outputs(self, t, state, inputs, params):
        return {}

    def flow_outputs(self, input_flows, params, ctx=None):
        return {}

    def rhs(self, t, state, inputs, params, signals=None):
        return jnp.zeros((0,))


def test_required_and_published_signal_names(asm1):
    """The declarative hooks the validation reads."""
    tank = CSTRUnit(name="tank", model=asm1, volume=1000.0,
                    input_port_names=["in"], conditions={"T": 293.15},
                    aeration=Aeration(do_setpoint=2.0, controller="do"))
    assert tank.required_signals == ("_aer_do_kla",)
    assert _do_controller(asm1, "do_signal").signal_names == ("do_signal",)
    # An uncontrolled (open-loop or anoxic) tank requires nothing.
    plain = CSTRUnit(name="t", model=asm1, volume=1.0, input_port_names=["in"],
                     conditions={"T": 293.15}, aeration=Aeration(kla=240.0))
    assert plain.required_signals == ()


def test_closed_loop_aeration_auto_wires_controller(asm1):
    """A closed-loop-aeration tank gets its controller (and sensor tap) created
    by the plant, so the bus validates and the controller is referenceable."""
    p = Plant(name="auto")
    p.add_unit(CSTRUnit(name="tank", model=asm1, volume=1000.0,
                        input_port_names=["in"], conditions={"T": 293.15},
                        aeration=Aeration(do_setpoint=2.0, controller="do")))
    p._build_state_layout()                # auto-wires + validates; must not raise
    assert "do" in p.units                 # controller named after the shared id
    assert p.units["do"].signal_names == ("_aer_do_kla",)


def test_missing_publisher_raises_clearly(asm1):
    """A (non-CSTR) consumer whose signal nobody publishes fails at setup, naming
    the unit and the missing signal -- not a deep KeyError."""
    p = Plant(name="bad")
    p.add_unit(_SignalConsumer("do_signal"))
    with pytest.raises(ValueError, match="consumes control signal 'do_signal'"):
        p._build_state_layout()


def test_typo_in_signal_name_raises(asm1):
    """A mismatch between the consumed and published name is caught."""
    p = Plant(name="typo")
    p.add_unit(_SignalConsumer("do_signl"))            # typo
    p.add_unit(_do_controller(asm1, "do_signal"))      # publishes the correct name
    with pytest.raises(ValueError, match="do_signl"):
        p._build_state_layout()


def test_duplicate_published_signal_name_raises(asm1):
    """Two controllers publishing the SAME signal name are rejected at setup: the
    bus is gathered by name (dict.update), so a duplicate would silently overwrite
    -- one controller's output discarded while its integral keeps winding."""
    p = Plant(name="dup")
    p.add_unit(_do_controller(asm1, "do_kla"))         # name 'do_ctrl'
    p.add_unit(PIController(
        name="do_ctrl2", model=asm1, measured_species="SO", setpoint=2.0,
        Kp=25.0, Ti=0.002, Tt=0.001, offset=120.0, out_min=0.0, out_max=360.0,
        signal_name="do_kla"))                         # same signal name, other unit
    with pytest.raises(ValueError, match="published by both"):
        p._build_state_layout()


def test_unknown_publisher_skips_validation(asm1):
    """If a signal *producer* doesn't declare signal_names, the published set is
    unknown, so validation is skipped rather than risk rejecting a valid plant."""
    class CustomController:
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
    p.add_unit(_SignalConsumer("do_signal"))
    p.add_unit(CustomController())
    p._build_state_layout()        # must not raise (publisher set is unknown)
