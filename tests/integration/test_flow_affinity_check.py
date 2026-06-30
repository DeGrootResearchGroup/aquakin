"""The recycle-flow affinity consistency check.

``Plant._recycle._resolve_flows`` resolves the recycle flows by treating the flow network
as an affine map and probing it at two points -- exact only if every unit's
``flow_outputs`` is affine in the recycle flows. A threshold-mode ``SplitterUnit``
(or a ``StorageTank`` bypass) is piecewise-linear, so a recycle-dependent inlet
that crosses its kink resolves the recycle flows inaccurately. The plant warns
when that happens; these tests pin the detection (precise, no false positives).
"""

import warnings

import jax.numpy as jnp
import pytest

import aquakin
from aquakin.plant.mixer import MixerUnit, SplitterUnit
from aquakin.plant.plant import Plant


@pytest.fixture(scope="module")
def net():
    return aquakin.load_network("asm1")


def _recycle_plant(net, threshold):
    """A minimal recycle: influent + (split remainder) -> mixer -> threshold
    split; the remainder feeds back, so the split's inlet depends on the recycle
    flow."""
    p = Plant("t")
    p.add_unit(MixerUnit("mix", ["feed_in", "recycle_in"], net))
    p.add_unit(SplitterUnit("split", net, threshold=threshold,
                            threshold_port="over", remainder_port="rem"))
    p.add_influent("feed", net.influent({}, Q=50.0), to="mix.feed_in")
    p.connect("mix.out", "split.in")
    p.connect("split.rem", "mix.recycle_in")        # recycle back-edge
    p._build_state_layout()
    p._build_parameter_layout()
    return p


def _check(plant, net):
    return plant._recycle._resolve_flows(jnp.asarray(0.0), net.default_parameters(),
                                check_affine=True)


def test_warn_helper_fires_on_inconsistency():
    p = Plant("h")
    with pytest.warns(UserWarning, match="non-affine"):
        p._warn_if_flow_nonaffine(jnp.asarray([10.0]), jnp.asarray([100.0]))


def test_warn_helper_silent_when_consistent():
    p = Plant("h")
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # any warning -> failure
        p._warn_if_flow_nonaffine(jnp.asarray([100.0]), jnp.asarray([100.0]))


def test_recycle_crossing_threshold_warns(net):
    """Threshold 50.5: the recycle drives the split's inlet across the kink, so
    the affine probe linearises across it and the solve is inconsistent."""
    plant = _recycle_plant(net, threshold=50.5)
    with pytest.warns(UserWarning, match="Recycle-flow solve is inconsistent"):
        _check(plant, net)


def test_recycle_below_threshold_is_silent(net):
    """A threshold the flow never reaches: the split stays in one linear piece,
    so the flow rule is affine over the operating range and there is no warning."""
    plant = _recycle_plant(net, threshold=1e9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check(plant, net)


def test_affine_recycle_plant_is_silent(net):
    """A ratio splitter (affine) in the same recycle topology never warns."""
    p = Plant("affine")
    p.add_unit(MixerUnit("mix", ["feed_in", "recycle_in"], net))
    p.add_unit(SplitterUnit("split", net, output_port_ratios={"out": 0.7, "rem": 0.3}))
    p.add_influent("feed", net.influent({}, Q=50.0), to="mix.feed_in")
    p.connect("mix.out", "split.in")
    p.connect("split.rem", "mix.recycle_in")
    p._build_state_layout()
    p._build_parameter_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check(p, net)
