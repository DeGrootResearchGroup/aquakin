"""The shared inlet-temperature heat-balance helper (``mixed_temperature``).

One rule combines inlet temperatures for every multi-inlet unit (mixer, CSTR,
clarifier, digester). The cases below pin the two behaviours that matter for a
temperature-carrying plant with recycle loops:

* a temperature-agnostic (or zero-flow recycle-seed) inlet is *ignored*, not
  allowed to force the whole mix to ``None`` (which silently disabled
  temperature propagation around a seeded recycle loop);
* the flow-weighting is zero-flow-safe -- a momentarily all-zero-flow mix uses
  the plain mean, not a divide-by-~0 that collapses the result toward 0 K.
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.mixer import MixerUnit
from aquakin.plant.streams import Stream, mixed_temperature


def _net():
    return aquakin.load_model("asm1")


def _stream(net, Q, T):
    return Stream(Q=jnp.asarray(float(Q)), C=net.default_concentrations(),
                  model=net, T=None if T is None else jnp.asarray(float(T)))


def test_flow_weighted_when_all_inlets_carry_temperature():
    net = _net()
    inputs = {"a": _stream(net, 3.0, 280.0), "b": _stream(net, 1.0, 300.0)}
    T = mixed_temperature(inputs, ["a", "b"])
    assert float(T) == pytest.approx((3.0 * 280.0 + 1.0 * 300.0) / 4.0)


def test_none_inlet_is_ignored_not_poisoning():
    """A temperature-agnostic / zero-flow-seed inlet (T=None) is dropped, so the
    temperature-carrying inlet still sets the result (regression for the seeded
    recycle that disabled temperature propagation around the loop)."""
    net = _net()
    inputs = {"fresh": _stream(net, 5.0, 290.0), "seed": _stream(net, 0.0, None)}
    T = mixed_temperature(inputs, ["fresh", "seed"])
    assert T is not None
    assert float(T) == pytest.approx(290.0)


def test_all_agnostic_returns_none():
    net = _net()
    inputs = {"a": _stream(net, 1.0, None), "b": _stream(net, 2.0, None)}
    assert mixed_temperature(inputs, ["a", "b"]) is None


def test_zero_flow_carriers_fall_back_to_mean_not_zero_kelvin():
    """All temperature-carrying inlets at zero flow -> plain mean, not ~0 K from
    dividing by the flow epsilon."""
    net = _net()
    inputs = {"a": _stream(net, 0.0, 280.0), "b": _stream(net, 0.0, 300.0)}
    T = mixed_temperature(inputs, ["a", "b"])
    assert float(T) == pytest.approx(290.0)   # mean, emphatically not ~0


def test_single_inlet_returns_its_temperature():
    net = _net()
    inputs = {"only": _stream(net, 7.0, 295.0)}
    assert float(mixed_temperature(inputs, ["only"])) == pytest.approx(295.0)


def test_mixer_ignores_zero_flow_agnostic_seed_inlet():
    """At the unit level: a MixerUnit fed a temperature-carrying influent plus a
    zero-flow agnostic recycle seed emits the influent temperature (the seeded
    recycle no longer poisons the heat balance)."""
    net = _net()
    mix = MixerUnit("mix", ["fresh", "recycle"], net)
    inputs = {"fresh": _stream(net, 10.0, 288.0), "recycle": _stream(net, 0.0, None)}
    out = mix.compute_outputs(jnp.asarray(0.0), jnp.zeros((0,)), inputs,
                              net.default_parameters())["out"]
    assert out.T is not None
    assert float(out.T) == pytest.approx(288.0)
