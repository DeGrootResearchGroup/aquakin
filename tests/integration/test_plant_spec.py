"""``Plant.to_spec`` / ``Plant.from_spec``: a plant as plain data and back.

The central check is that a rebuilt plant is the same plant: the same spec, the
same initial state, and a bit-identical right-hand side at that state.
"""

import json
import shutil
from pathlib import Path

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
from aquakin.plant.bsm import bsm2_constant_influent, build_bsm1, build_bsm2, load_bsm1_influent
from aquakin.plant.bsm.bsm2 import BSM2_ASM1_PARAMETERS, BSM2_CONSTANT_INFLUENT_T, bsm2_asm1_model
from aquakin.plant.spec import SpecError, SpecWarning, load_spec_file
from aquakin.plant.streams import Stream


def _asm1():
    # Loaded per use, not held at module level: other tests clear the model
    # cache, after which a held model is a different object from a fresh load.
    return aquakin.load_model("asm1")


def _adm1():
    return aquakin.load_model("adm1")


BSM2_PROJECT = Path(__file__).parents[2] / "examples" / "projects" / "bsm2"
FEED = dict(
    SI=30.0, SS=69.5, XI=51.2, XS=202.32, XB_H=28.17, SNH=31.56, SND=6.95, XND=10.59, SALK=7.0
)  # fmt: skip


def _bsm1(**kw):
    plant = build_bsm1(**kw)
    plant.add_influent(
        "feed", InfluentSeries.constant(_asm1(), Q=18446.0, **FEED), to=plant.influent_endpoint
    )
    return plant


def _two_model_plant():
    """A small ASM1 line with a dosed reagent, an ADM1 digester behind two
    translators, and a temperature-carrying seed on the reject recycle."""
    p = Plant("two-model")
    p.add_unit(MixerUnit("mix", ["fresh", "reject"], _asm1()))
    p.add_unit(
        DosingUnit("carbon", Reagent.from_species(_asm1(), SS=400000.0, label="carbon"), flow=2.0)
    )
    p.add_unit(
        CSTRUnit(
            "tank", _asm1(), 1000.0, ["in"], conditions={"T": 293.15}, aeration=Aeration(kla=120.0)
        )
    )
    p.add_unit(RatioSplitter("split", _asm1(), {"to_digester": 0.02, "effluent": 0.98}))
    p.add_unit(ADM1DigesterUnit("digester", _adm1(), 3400.0, conditions={"T": 308.15}))
    p.add_influent(
        "feed", InfluentSeries.constant(_asm1(), Q=18446.0, T=288.15, **FEED), to="mix.fresh"
    )
    p.connect("mix", "carbon.in")
    p.connect("carbon.out", "tank")
    p.connect("tank", "split")
    p.connect("split.to_digester", "digester", translator=ASM1toADM1(_asm1(), _adm1()))
    seed = Stream(
        Q=np.float64(0.0),
        C=_asm1().default_concentrations(),
        model=_asm1(),
        scalars={"T": np.float64(308.15)},
    )
    p.connect(
        "digester",
        "mix.reject",
        translator=ADM1toASM1(_adm1(), _asm1(), pH_adm=7.2),
        initial_value=seed,
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


def test_every_setting_is_written_with_the_aquakin_version():
    spec = _bsm1(use_takacs=True).to_spec()
    assert spec["format"] == 2 and spec["aquakin"] == aquakin.__version__
    clarifier = spec["units"]["clarifier"]
    # Defaults are written too, so a later change of default cannot change the plant.
    assert clarifier["interface_flux"] == "godunov" and clarifier["n_layers"] == 10
    assert spec["units"]["tank3"]["aeration"]["Kp"] == 25.0
    assert spec["options"] == {"recycle_passes": 3, "recycle_tol": 1e-8, "recycle_max_passes": 100}
    assert (
        spec["models"]["asm1"]["source"] == "shipped"
        and len(spec["models"]["asm1"]["sha256"]) == 64
    )


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
    assert reject["translator"]["kind"] == "adm1_to_asm1" and reject["translator"]["pH_adm"] == 7.2
    assert reject["seed"]["Q"] == 0.0 and reject["seed"]["T"] == 308.15
    assert spec["units"]["carbon"]["reagent"] == {
        "model": "asm1",
        "label": "carbon",
        "composition": {"SS": 400000.0},
    }
    assert spec["units"]["feed"]["T"] == 288.15
    _assert_same_plant(p, Plant.from_spec(spec))


def test_labels_are_display_text_beside_stable_names():
    p = _bsm1()
    p.labels["tank3"] = "Aerobic zone 1"
    spec = p.to_spec()
    assert spec["units"]["tank3"]["label"] == "Aerobic zone 1"
    q = Plant.from_spec(spec)
    assert q.labels == {"tank3": "Aerobic zone 1"} and "tank3" in q.units
    _assert_same_plant(p, q)


def test_derived_controllers_are_not_written():
    p = Plant("closed-loop")
    aeration = Aeration(do_setpoint=2.0)
    p.add_unit(
        CSTRUnit("tank", _asm1(), 1333.0, ["in"], conditions={"T": 293.15}, aeration=aeration)
    )
    p.add_influent("feed", InfluentSeries.constant(_asm1(), Q=18446.0, **FEED), to="tank.in")
    y = p.initial_state()  # building the layout derives the DO controller
    assert p._derived_units == {"tank_aeration"}
    spec = p.to_spec()
    assert set(spec["units"]) == {"feed", "tank"}
    assert spec["connections"] == ["feed.out -> tank.in"]
    assert spec["units"]["tank"]["aeration"]["do_setpoint"] == 2.0
    q = Plant.from_spec(spec)
    np.testing.assert_array_equal(np.asarray(q.derivative(y)), np.asarray(p.derivative(y)))
    assert q._derived_units == {"tank_aeration"}


def test_editor_layout_is_accepted_and_ignored():
    spec = _bsm1().to_spec() | {"layout": {"tank1": [0, 0]}}
    assert "layout" not in Plant.from_spec(spec).to_spec()


def test_the_bsm2_project_is_bsm2():
    """The shipped BSM2 project (a 15 degC ASM1 model file + flowsheet) rebuilds
    the plant build_bsm2 makes, run with the BSM2 parameter set."""
    q = Plant.from_spec(BSM2_PROJECT / "flowsheet.yaml")
    m15 = bsm2_asm1_model()
    ref = build_bsm2(asm1_model=m15)
    ref.add_influent(
        "influent",
        bsm2_constant_influent(m15, T=BSM2_CONSTANT_INFLUENT_T),
        to=ref.influent_endpoint,
    )
    params = ref.parameter_values({f"asm1.{k}": v for k, v in BSM2_ASM1_PARAMETERS.items()})
    np.testing.assert_array_equal(np.asarray(q.default_parameters()), np.asarray(params))
    y = ref.initial_state()
    np.testing.assert_array_equal(
        np.asarray(q.derivative(y)), np.asarray(ref.derivative(y, params))
    )
    assert Plant.from_spec(BSM2_PROJECT / "flowsheet.yaml").to_spec() == q.to_spec()


def test_a_changed_model_file_warns(tmp_path):
    shutil.copytree(BSM2_PROJECT, tmp_path / "bsm2")
    model = tmp_path / "bsm2" / "models" / "asm1_bsm2.yaml"
    model.write_text(model.read_text().replace("muH: {value: 4.0", "muH: {value: 4.5"))
    with pytest.warns(SpecWarning, match="model 'asm1_bsm2'.*has changed"):
        Plant.from_spec(tmp_path / "bsm2" / "flowsheet.yaml")


def test_what_a_spec_cannot_express_is_refused_by_name():
    with pytest.raises(SpecError, match="model 'asm1' is neither a shipped model"):
        build_bsm2().to_spec()
    p = build_bsm1()
    p.add_influent("feed", load_bsm1_influent("dry", _asm1()), to=p.influent_endpoint)
    with pytest.raises(
        SpecError, match=r"influent 'feed' varies in time; write the spec to a file"
    ):
        p.to_spec()
    p = Plant("delay")
    p.add_unit(
        HydraulicDelayUnit(
            "d", _asm1(), 0.1, initial_concentrations=_asm1().default_concentrations()
        )
    )
    with pytest.raises(SpecError, match="'initial_concentrations'"):
        p.to_spec()


def _units(change):
    def apply(s):
        units = dict(s["units"])
        change(units)
        return {**s, "units": units}

    return apply


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s | {"extra": 1}, "extra: Extra inputs are not permitted"),
        (lambda s: s | {"format": 1}, "spec format 1 is not supported"),
        (lambda s: s | {"options": {"tol": 1}}, "options.tol: Extra inputs"),
        (_units(lambda u: u.update(tank1={"kind": "vat"})), "units.tank1: KeyError"),
        (_units(lambda u: u.update(tank1={"kind": "cstr", "model": "asm1"})), "units.tank1: .*'volume' is required"),
        (_units(lambda u: u.update(tank1={**u["tank1"], "aeraton": {"kla": 240.0}})), "'aeraton' is not a setting of cstr. Did you mean: aeration"),
        (_units(lambda u: u.update(tank1={**u["tank1"], "volume": "1000"})), "'volume' must be a real number"),
        (_units(lambda u: u.update(tank1={**u["tank1"], "model": "asm3"})), "model 'asm3' is not listed under 'models'"),
        (_units(lambda u: u.update({"tank.9": u["tank1"]})), "unit identifier 'tank.9'"),
        (lambda s: {**s, "connections": [*s["connections"], "tank1.out tank2.in"]}, r"connections\[11\]: write a connection"),
        (lambda s: {**s, "connections": [*s["connections"], "nowhere.out -> tank1.inlet"]}, r"connections\[11\]"),
        (lambda s: {**s, "connections": s["connections"][:-1]}, "influent 'feed' must feed exactly one"),
    ],
)  # fmt: skip
def test_a_bad_spec_says_where(change, message):
    with pytest.raises(SpecError, match=message):
        Plant.from_spec(change(_bsm1().to_spec()))


def test_yaml_is_read_with_yaml_1_2_scalars(tmp_path):
    spec = _bsm1().to_spec()
    text = yaml.safe_dump(spec, sort_keys=False)
    # Hand-edited forms YAML 1.1 would misread: 1e-8 as text, a port named on as True.
    text = text.replace("recycle_tol: 1.0e-08", "recycle_tol: 1e-8")
    text = text.replace("internal_recycle: 55338.0", "on: 55338.0")
    text = text.replace("tank5_split.internal_recycle", "tank5_split.on")
    path = tmp_path / "hand.yaml"
    path.write_text(text)
    data = load_spec_file(path)
    assert data["options"]["recycle_tol"] == 1e-8
    assert "on" in data["units"]["tank5_split"]["output_port_flows"]
    q = Plant.from_spec(path)
    assert q.units["tank5_split"].output_ports == ["on", "to_clarifier"]


def test_unit_names_the_plant_cannot_address_are_refused():
    from aquakin.plant import SetpointSplitter, WiringError

    for unit in (
        SetpointSplitter("asm1", _asm1(), {"muH": 1.0}, "rest"),  # would shadow asm1.muH
        CSTRUnit("tank.1", _asm1(), 1.0, ["in"], conditions={"T": 293.15}),
        CSTRUnit("tank", _asm1(), 1.0, ["in put"], conditions={"T": 293.15}),
    ):
        with pytest.raises(WiringError):
            Plant("x").add_unit(unit)
    with pytest.raises(WiringError):
        Plant("x").add_influent("feed.1", InfluentSeries.constant(_asm1(), Q=1.0))


def test_a_plant_built_before_the_model_cache_was_cleared_can_still_be_written():
    p = _bsm1()
    aquakin.clear_model_cache()
    _assert_same_plant(p, Plant.from_spec(p.to_spec()))


def test_a_modified_model_is_refused_and_a_model_file_is_found_by_itself(tmp_path):
    with pytest.raises(SpecError, match="neither a shipped model"):
        p = build_bsm2(asm1_model=bsm2_asm1_model())  # dataclasses.replace of ASM1
        p.to_spec()
    m15 = aquakin.load_model_from_file(BSM2_PROJECT / "models" / "asm1_bsm2.yaml")
    p = build_bsm2(asm1_model=m15)
    spec = p.to_spec(tmp_path / "flowsheet.yaml")  # no models= needed: the file is recorded
    assert spec["models"]["asm1_bsm2"]["source"].endswith("models/asm1_bsm2.yaml")
    assert Plant.from_spec(tmp_path / "flowsheet.yaml").to_spec() == Plant.from_spec(spec).to_spec()


def _dynamic_bsm1():
    p = build_bsm1()
    p.add_influent("feed", load_bsm1_influent("dry", _asm1()), to=p.influent_endpoint)
    return p


def test_a_time_varying_influent_is_written_as_a_table_beside_the_spec(tmp_path):
    p = _dynamic_bsm1()
    spec = p.to_spec(tmp_path / "flowsheet.yaml")
    feed = spec["units"]["feed"]
    assert (
        feed["series"] == "data/feed.csv" and feed["Q"] is None and len(feed["series_sha256"]) == 64
    )
    assert (tmp_path / "data" / "feed.csv").read_text().startswith("t,Q,SI,SS,")
    q = Plant.from_spec(tmp_path / "flowsheet.yaml")
    a, b = p.influents["feed"], q.influents["feed"]
    for x, y in ((a.t, b.t), (a.Q, b.Q), (a.C, b.C)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    y0 = p.initial_state()
    for t in (0.0, 0.37):
        np.testing.assert_array_equal(
            np.asarray(q.derivative(y0, t=t)), np.asarray(p.derivative(y0, t=t))
        )
    # Written again, the plant refers to the same table rather than copying it.
    assert q.to_spec(tmp_path / "flowsheet.yaml") == spec


def test_a_changed_influent_table_warns(tmp_path):
    _dynamic_bsm1().to_spec(tmp_path / "flowsheet.yaml")
    table = tmp_path / "data" / "feed.csv"
    lines = table.read_text().splitlines()
    lines[1] = lines[1].replace(lines[1].split(",")[1], "1.0", 1)
    table.write_text("\n".join(lines) + "\n")
    with pytest.warns(SpecWarning, match="influent 'feed'.*has changed"):
        Plant.from_spec(tmp_path / "flowsheet.yaml")


def test_an_influent_table_reads_back_exactly(tmp_path):
    from aquakin.plant import read_influent_table, write_influent_table

    series = InfluentSeries.constant(_asm1(), Q=0.1 + 0.2, T=288.15, SS=1 / 3)
    path = write_influent_table(series, tmp_path / "inf.csv")
    back = read_influent_table(path, _asm1())
    for x, y in ((series.t, back.t), (series.Q, back.Q), (series.C, back.C), (series.T, back.T)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    (tmp_path / "bad.csv").write_text("t,Q,XX\n0,1,2\n1,1,2\n")
    with pytest.raises(ValueError, match=r"column\(s\) \['XX'\] are not species"):
        read_influent_table(tmp_path / "bad.csv", _asm1())
