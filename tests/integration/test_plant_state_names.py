"""Every unit names its state entries, and a plant state can be stored by name.

Stored results key a plant state as ``{unit: {state name: value}}``, so they
survive a change of unit order; this pins that every shipped unit names each of
its entries once, and that the by-name form reads back exactly.
"""

import numpy as np
import pytest

import aquakin
import aquakin.plant as plant
from aquakin.plant import (
    HeatBalanceTemperature,
    InfluentSeries,
    InterfaceSettling,
    LayeredSettling,
    Plant,
    SBRPhase,
    SBRUnit,
)
from aquakin.plant.takacs import TakacsClarifier
from aquakin.plant.bsm import build_bsm1, build_bsm2
from aquakin.plant.catalog import build_unit


def _buildable():
    """The catalogue test's settings that build each kind."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "unit" / "test_plant_catalog.py"
    spec = importlib.util.spec_from_file_location("_catalog_settings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BUILDABLE


BUILDABLE = _buildable()


def _check(unit):
    names = tuple(unit.state_names())
    assert len(names) == unit.state_size, type(unit).__name__
    assert len(set(names)) == len(names), type(unit).__name__
    return names


@pytest.mark.parametrize("kind", sorted(set(BUILDABLE) - {"influent"}))
def test_every_kind_names_each_state_entry_once(kind):
    _check(build_unit(kind, "u", BUILDABLE[kind]))


def test_layered_units_name_their_layers():
    m = aquakin.load_model("asm1")
    ifas = build_unit("ifas", "u", {**BUILDABLE["ifas"], "n_layers": 3})
    names = _check(ifas)
    assert names[: m.n_species] == tuple(m.species)
    assert names[m.n_species] == "film1.SI" and names[-1] == "film3.SALK"

    for mode, holdup in (("per_species", False), ("per_species", True), ("lumped_tss", True)):
        settler = TakacsClarifier(
            "s",
            m,
            1500.0,
            4.0,
            underflow_Q=18000.0,
            composition_mode=mode,
            soluble_holdup=holdup,
        )
        names = _check(settler)
        assert names[0] == ("layer1.TSS" if mode == "lumped_tss" else "layer1.XS")

    for settling, extra in (
        (InterfaceSettling(1.0, 10.0), ("clarity",)),
        (LayeredSettling(n_layers=3), ("layer1.ratio", "layer2.ratio", "layer3.ratio")),
    ):
        sbr = SBRUnit(
            "s",
            m,
            [SBRPhase("fill", 0.1, feed=True)],
            100.0,
            1.0,
            1.0,
            settling,
            conditions={"T": 293.15},
        )
        assert _check(sbr)[m.n_species :] == ("V", *extra)


def test_a_plant_state_reads_back_by_name_in_another_unit_order():
    m = aquakin.load_model("asm1")
    p = build_bsm1(use_takacs=True)
    p.add_influent("feed", InfluentSeries.constant(m, Q=18446.0, SS=60.0), to=p.influent_endpoint)
    y = np.asarray(p.initial_state()) * 1.01 + 0.5
    named = p.named_state(y)
    assert set(named) == {"tank1", "tank2", "tank3", "tank4", "tank5", "clarifier"}
    assert named["tank3"]["SNH"] == pytest.approx(y[p._state_layout["tank3"][0] + 9])
    np.testing.assert_array_equal(np.asarray(p.state_from_named(named)), y)

    # The same plant with its units listed in reverse order has another state
    # layout (the settler comes first), but the named state lands on the same
    # entries.
    spec = p.to_spec()
    q = Plant.from_spec(dict(spec, units=dict(reversed(list(spec["units"].items())))))
    assert q.named_state(q.state_from_named(named)) == named
    assert q._state_layout["clarifier"][0] != p._state_layout["clarifier"][0]


def test_partial_and_misspelled_named_states():
    p = build_bsm1()
    y0 = np.asarray(p.initial_state())
    one = np.asarray(p.state_from_named({"tank2": {"SO": 1.5}}))
    changed = np.flatnonzero(one != y0)
    assert changed.tolist() == [p._state_layout["tank2"][0] + 7] and one[changed[0]] == 1.5
    with pytest.raises(KeyError, match="has no state entry 'SNHH'.*Did you mean: SNH"):
        p.state_from_named({"tank2": {"SNHH": 1.0}})
    with pytest.raises(plant.UnknownUnitError, match="no unit with a state named 'tank9'"):
        p.state_from_named({"tank9": {"SO": 1.0}})


def test_bsm2_with_a_heat_balance_names_every_entry():
    p = build_bsm2(temperature_model=HeatBalanceTemperature())
    asm1 = p.units["tank1"].model
    p.add_influent(
        "feed",
        InfluentSeries.constant(asm1, Q=20648.0, T=288.15, SS=60.0),
        to=p.influent_endpoint,
    )
    names = p.state_names()
    assert sum(len(v) for v in names.values()) == p.initial_state().shape[0]
    assert names["tank1"][-1] == "T" and "T" not in names["digester"]
    y = np.asarray(p.initial_state())
    np.testing.assert_array_equal(np.asarray(p.state_from_named(p.named_state(y))), y)
