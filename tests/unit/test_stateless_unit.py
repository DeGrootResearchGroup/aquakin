"""The StatelessUnit mixin: the shared no-state members for stateless units."""

import jax.numpy as jnp

import aquakin
from aquakin.plant.clarifier import IdealClarifier
from aquakin.plant.mixer import MixerUnit, RatioSplitter
from aquakin.plant.separators import IdealThickener
from aquakin.plant.units import StatelessUnit, Unit


def _net():
    return aquakin.load_model("asm1")


def test_stateless_units_inherit_the_mixin():
    net = _net()
    units = [
        MixerUnit("mix", ["a", "b"], net),
        RatioSplitter("split", net, output_port_ratios={"x": 0.5, "y": 0.5}),
        IdealClarifier(name="clar", model=net, underflow_Q=100.0),
        IdealThickener(name="thick", model=net, target_tss_percent=5.0),
    ]
    for u in units:
        assert isinstance(u, StatelessUnit)
        assert isinstance(u, Unit)               # still satisfies the contract


def test_mixin_provides_trivial_state_members():
    net = _net()
    m = MixerUnit("mix", ["a", "b"], net)
    assert m.state_size == 0
    assert m.initial_state().shape == (0,)
    # rhs is a no-op of the right (empty) shape, with the uniform signature.
    d = m.rhs(0.0, jnp.zeros((0,)), {}, jnp.zeros((0,)), None)
    assert d.shape == (0,)


def test_mixin_composes_with_dataclass_fields():
    """The mixin must not disturb the @dataclass fields of the subclass."""
    net = _net()
    m = MixerUnit("mix", ["a", "b"], net)
    assert m.name == "mix"
    assert m.input_port_names == ["a", "b"]
    assert m.model is net
