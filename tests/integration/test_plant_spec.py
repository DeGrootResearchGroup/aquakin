"""``Plant.to_spec`` / ``Plant.from_spec``: a plant as plain data and back.

The central check is that a rebuilt plant is the same plant: the same spec, the
same initial state, and a bit-identical right-hand side at that state.
"""

import json

import numpy as np
import pytest
import yaml

import aquakin
from aquakin.plant import (
    ADM1DigesterUnit,
    ADM1toASM1,
    Aeration,
    ASM1toADM1,
    CSTRUnit,
    DosingUnit,
    HydraulicDelayUnit,
    InfluentSeries,
    MixerUnit,
    Plant,
    RatioSplitter,
    Reagent,
)
from aquakin.plant.bsm import build_bsm1, build_bsm2, load_bsm1_influent
from aquakin.plant.spec import SpecError
from aquakin.plant.streams import Stream

ASM1 = aquakin.load_model("asm1")
ADM1 = aquakin.load_model("adm1")
FEED = dict(
    SI=30.0, SS=69.5, XI=51.2, XS=202.32, XB_H=28.17, SNH=31.56, SND=6.95, XND=10.59, SALK=7.0
)  # fmt: skip


def _bsm1(**kw):
    plant = build_bsm1(**kw)
    plant.add_influent(
        "feed", InfluentSeries.constant(ASM1, Q=18446.0, **FEED), to=plant.influent_endpoint
    )
    return plant


def _two_model_plant():
    """A small ASM1 line with a dosed reagent, an ADM1 digester behind two
    translators, and a temperature-carrying seed on the reject recycle."""
    p = Plant("two-model")
    p.add_unit(MixerUnit("mix", ["fresh", "reject"], ASM1))
    p.add_unit(
        DosingUnit("carbon", Reagent.from_species(ASM1, SS=400000.0, label="carbon"), flow=2.0)
    )
    p.add_unit(
        CSTRUnit(
            "tank", ASM1, 1000.0, ["in"], conditions={"T": 293.15}, aeration=Aeration(kla=120.0)
        )
    )
    p.add_unit(RatioSplitter("split", ASM1, {"to_digester": 0.02, "effluent": 0.98}))
    p.add_unit(ADM1DigesterUnit("digester", ADM1, 3400.0, conditions={"T": 308.15}))
    p.add_influent(
        "feed", InfluentSeries.constant(ASM1, Q=18446.0, T=288.15, **FEED), to="mix.fresh"
    )
    p.connect("mix", "carbon.in")
    p.connect("carbon.out", "tank")
    p.connect("tank", "split")
    p.connect("split.to_digester", "digester", translator=ASM1toADM1(ASM1, ADM1))
    seed = Stream(
        Q=np.float64(0.0),
        C=ASM1.default_concentrations(),
        model=ASM1,
        scalars={"T": np.float64(308.15)},
    )
    p.connect(
        "digester", "mix.reject", translator=ADM1toASM1(ADM1, ASM1, pH_adm=7.2), initial_value=seed
    )
    p.register_stream("effluent", "split.effluent")
    return p


def _assert_same_plant(p, q):
    assert q.to_spec() == p.to_spec()
    y = p.initial_state()
    np.testing.assert_array_equal(np.asarray(q.initial_state()), np.asarray(y))
    np.testing.assert_array_equal(np.asarray(q.derivative(y)), np.asarray(p.derivative(y)))


@pytest.mark.parametrize("use_takacs", [False, True])
def test_bsm1_round_trips_exactly(use_takacs):
    p = _bsm1(use_takacs=use_takacs)
    q = Plant.from_spec(p.to_spec())
    _assert_same_plant(p, q)
    assert q.list_streams() == p.list_streams()
    assert (q.influent_endpoint, q.effluent_endpoint) == (p.influent_endpoint, p.effluent_endpoint)


def test_a_spec_reads_back_from_yaml_and_json_files(tmp_path):
    p = _bsm1()
    for name in ("bsm1.yaml", "bsm1.json"):
        path = tmp_path / name
        p.to_spec(path)
        _assert_same_plant(p, Plant.from_spec(path))
    text = (tmp_path / "bsm1.yaml").read_text()
    assert "kind: cstr" in text and "inlet_mix.out -> tank1.inlet" in text
    json.loads((tmp_path / "bsm1.json").read_text())


def test_translators_seeds_reagents_and_influent_temperature_round_trip():
    p = _two_model_plant()
    spec = p.to_spec()
    reject = next(c for c in spec["connections"] if isinstance(c, dict) and c["to"] == "mix.reject")
    assert reject["translator"] == {"kind": "adm1_to_asm1", "pH_adm": 7.2}
    assert reject["seed"] == {"Q": 0.0, "T": 308.15}
    assert spec["units"]["carbon"]["reagent"] == {
        "model": "asm1",
        "label": "carbon",
        "composition": {"SS": 400000.0},
    }
    assert spec["units"]["feed"]["T"] == 288.15
    _assert_same_plant(p, Plant.from_spec(spec))


def test_derived_controllers_are_not_written():
    p = Plant("closed-loop")
    tank = CSTRUnit(
        "tank", ASM1, 1333.0, ["in"], conditions={"T": 293.15}, aeration=Aeration(do_setpoint=2.0)
    )
    p.add_unit(tank)
    p.add_influent("feed", InfluentSeries.constant(ASM1, Q=18446.0, **FEED), to="tank.in")
    y = p.initial_state()  # building the layout derives the DO controller
    assert p._derived_units == {"tank_aeration"}
    spec = p.to_spec()
    assert set(spec["units"]) == {"feed", "tank"}
    assert spec["connections"] == ["feed.out -> tank.in"]
    assert spec["units"]["tank"]["aeration"] == {"do_setpoint": 2.0}
    q = Plant.from_spec(spec)
    np.testing.assert_array_equal(np.asarray(q.derivative(y)), np.asarray(p.derivative(y)))
    assert q._derived_units == {"tank_aeration"}


def test_editor_layout_is_accepted_and_ignored():
    spec = _bsm1().to_spec() | {"layout": {"tank1": [0, 0]}}
    assert "layout" not in Plant.from_spec(spec).to_spec()


def test_what_a_spec_cannot_express_is_refused_by_name():
    with pytest.raises(SpecError, match="model 'asm1' is not a shipped model"):
        build_bsm2().to_spec()
    p = build_bsm1()
    p.add_influent("feed", load_bsm1_influent("dry", ASM1), to=p.influent_endpoint)
    with pytest.raises(SpecError, match="influent 'feed' varies in time"):
        p.to_spec()
    p = Plant("delay")
    p.add_unit(
        HydraulicDelayUnit("d", ASM1, 0.1, initial_concentrations=ASM1.default_concentrations())
    )
    with pytest.raises(SpecError, match="'initial_concentrations'"):
        p.to_spec()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s | {"extra": 1}, "unknown top-level key"),
        (lambda s: s | {"format": 2}, "spec format 2 is not supported"),
        (lambda s: s | {"options": {"tol": 1}}, "options: unknown key"),
        (lambda s: {**s, "units": {**s["units"], "tank1": {"kind": "vat"}}}, "units.tank1: KeyError"),
        (lambda s: {**s, "units": {**s["units"], "tank1": {"kind": "cstr", "model": "asm1"}}}, "units.tank1: .*'volume' is required"),
        (lambda s: {**s, "connections": [*s["connections"], "tank1.out tank2.in"]}, r"connections\[11\]: write a connection"),
        (lambda s: {**s, "connections": [*s["connections"], "nowhere.out -> tank1.inlet"]}, r"connections\[11\]"),
        (lambda s: {**s, "connections": s["connections"][:-1]}, "influent 'feed' must feed exactly one"),
    ],
)  # fmt: skip
def test_a_bad_spec_says_where(change, message):
    with pytest.raises(SpecError, match=message):
        Plant.from_spec(change(_bsm1().to_spec()))


def test_the_spec_is_plain_yaml():
    text = yaml.safe_dump(_bsm1(use_takacs=True).to_spec())
    assert Plant.from_spec(yaml.safe_load(text)).to_spec() == _bsm1(use_takacs=True).to_spec()
