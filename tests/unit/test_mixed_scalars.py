"""The shared inlet side-channel combiner (``mixed_scalars``).

One rule combines the inlet side-channel scalars (temperature, indicator
density, ...) for every multi-inlet unit (mixer, CSTR, clarifier, digester). The
cases below pin the two behaviours that matter for a temperature-carrying plant
with recycle loops (exercised through the ``"T"`` scalar):

* an agnostic (or zero-flow recycle-seed) inlet -- one that does not carry the
  scalar -- is *ignored*, not allowed to force the whole mix to drop it (which
  silently disabled temperature propagation around a seeded recycle loop);
* the flow-weighting is zero-flow-safe -- a momentarily all-zero-flow mix uses
  the plain mean, not a divide-by-~0 that collapses the result toward 0 K.

A scalar that no inlet carries is omitted from the returned map entirely (the
static structural property the old per-scalar ``None`` return was).
"""

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.mixer import MixerUnit
from aquakin.plant.streams import Stream, make_scalars, mixed_scalars


def _net():
    return aquakin.load_model("asm1")


def _stream(net, Q, T):
    return Stream(Q=jnp.asarray(float(Q)), C=net.default_concentrations(),
                  model=net, scalars=make_scalars(T=None if T is None else jnp.asarray(float(T))))


def _mixed_T(inputs, names):
    """The temperature channel of the shared combiner (or ``None`` if none carry it)."""
    return mixed_scalars(inputs, names).get("T")


def test_flow_weighted_when_all_inlets_carry_temperature():
    net = _net()
    inputs = {"a": _stream(net, 3.0, 280.0), "b": _stream(net, 1.0, 300.0)}
    T = _mixed_T(inputs, ["a", "b"])
    assert float(T) == pytest.approx((3.0 * 280.0 + 1.0 * 300.0) / 4.0)


def test_none_inlet_is_ignored_not_poisoning():
    """An agnostic / zero-flow-seed inlet (no ``"T"``) is dropped, so the
    temperature-carrying inlet still sets the result (regression for the seeded
    recycle that disabled temperature propagation around the loop)."""
    net = _net()
    inputs = {"fresh": _stream(net, 5.0, 290.0), "seed": _stream(net, 0.0, None)}
    T = _mixed_T(inputs, ["fresh", "seed"])
    assert T is not None
    assert float(T) == pytest.approx(290.0)


def test_all_agnostic_omits_the_scalar():
    net = _net()
    inputs = {"a": _stream(net, 1.0, None), "b": _stream(net, 2.0, None)}
    combined = mixed_scalars(inputs, ["a", "b"])
    assert "T" not in combined  # no inlet carries it -> omitted entirely
    assert _mixed_T(inputs, ["a", "b"]) is None


def test_zero_flow_carriers_fall_back_to_mean_not_zero_kelvin():
    """All temperature-carrying inlets at zero flow -> plain mean, not ~0 K from
    dividing by the flow epsilon."""
    net = _net()
    inputs = {"a": _stream(net, 0.0, 280.0), "b": _stream(net, 0.0, 300.0)}
    T = _mixed_T(inputs, ["a", "b"])
    assert float(T) == pytest.approx(290.0)   # mean, emphatically not ~0


def test_single_inlet_returns_its_temperature():
    net = _net()
    inputs = {"only": _stream(net, 7.0, 295.0)}
    assert float(_mixed_T(inputs, ["only"])) == pytest.approx(295.0)


def test_make_scalars_drops_none_entries():
    """A leaf stream's side-channel map omits the agnostic (None) quantities, so
    absence -- not a stored None -- is what marks a scalar as not carried."""
    made = make_scalars(T=jnp.asarray(290.0), org=None)
    assert set(made) == {"T"} and float(made["T"]) == pytest.approx(290.0)
    assert make_scalars(T=None) == {}


def test_mixed_scalars_generalizes_to_an_arbitrary_scalar():
    """The combiner is not special-cased to T/org: any named scalar a stream
    carries in its map is flow-weighted the same way -- the point of the
    single-map data model. Here an ad-hoc 'ph' rides through with no new field,
    copier or combiner."""
    net = _net()
    C = net.default_concentrations()
    a = Stream(Q=jnp.asarray(3.0), C=C, model=net, scalars={"ph": jnp.asarray(7.0)})
    b = Stream(Q=jnp.asarray(1.0), C=C, model=net, scalars={"ph": jnp.asarray(8.0)})
    n = Stream(Q=jnp.asarray(2.0), C=C, model=net)  # carries no ph
    combined = mixed_scalars({"a": a, "b": b, "n": n}, ["a", "b", "n"], keys=("ph",))
    assert float(combined["ph"]) == pytest.approx((3.0 * 7.0 + 1.0 * 8.0) / 4.0)
    # the default keys only combine the first-class T/org, so 'ph' is not picked up
    assert "ph" not in mixed_scalars({"a": a, "b": b}, ["a", "b"])


def test_mixer_ignores_zero_flow_agnostic_seed_inlet():
    """At the unit level: a MixerUnit fed a temperature-carrying influent plus a
    zero-flow agnostic recycle seed emits the influent temperature (the seeded
    recycle no longer poisons the heat balance)."""
    net = _net()
    mix = MixerUnit("mix", ["fresh", "recycle"], net)
    inputs = {"fresh": _stream(net, 10.0, 288.0), "recycle": _stream(net, 0.0, None)}
    out = mix.compute_outputs(jnp.asarray(0.0), jnp.zeros((0,)), inputs,
                              net.default_parameters())["out"]
    assert out.scalars.get("T") is not None
    assert float(out.scalars["T"]) == pytest.approx(288.0)
