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
    assert spec["format"] == 2 and spec["aquakin"].startswith(aquakin.__version__)
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
    p.add_influent(
        "influent", bsm2_constant_influent(m15, T=BSM2_CONSTANT_INFLUENT_T), to=p.influent_endpoint
    )
    # The model file sits outside the new project, so it is copied in beside it.
    spec = p.to_spec(tmp_path / "flowsheet.yaml")
    assert spec["models"]["asm1_bsm2"]["source"] == "models/asm1_bsm2.yaml"
    assert (tmp_path / "models" / "asm1_bsm2.yaml").read_text() == (
        BSM2_PROJECT / "models" / "asm1_bsm2.yaml"
    ).read_text()
    q = Plant.from_spec(tmp_path / "flowsheet.yaml")
    y = p.initial_state()
    np.testing.assert_array_equal(np.asarray(q.derivative(y)), np.asarray(p.derivative(y)))


def _dynamic_bsm1():
    p = build_bsm1()
    p.add_influent("feed", load_bsm1_influent("dry", _asm1()), to=p.influent_endpoint)
    return p


def test_a_time_varying_influent_is_written_as_a_table_beside_the_spec(tmp_path):
    p = _dynamic_bsm1()
    spec = p.to_spec(tmp_path / "flowsheet.yaml")
    feed = spec["units"]["feed"]
    assert feed["series"] == "data/feed.csv" and feed["Q"] is None
    assert len(spec["files"]["data/feed.csv"]) == 64
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
    with pytest.warns(SpecWarning, match="data/feed.csv has changed"):
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


def test_the_fingerprint_follows_what_the_plant_is_not_how_it_is_drawn(tmp_path):
    from aquakin.plant.spec import spec_fingerprint

    shutil.copytree(BSM2_PROJECT, tmp_path / "bsm2")
    path = tmp_path / "bsm2" / "flowsheet.yaml"
    base = spec_fingerprint(path)
    spec = load_spec_file(path)
    same = dict(spec, name="renamed", layout={"tank1": [0, 0]})
    same["units"] = {**spec["units"], "tank1": {**spec["units"]["tank1"], "label": "Zone 1"}}
    assert spec_fingerprint(same, base_dir=path.parent) == base
    # A model file moved elsewhere with the same content keeps it too.
    moved = tmp_path / "elsewhere" / "asm1_bsm2.yaml"
    moved.parent.mkdir()
    shutil.copy(tmp_path / "bsm2" / "models" / "asm1_bsm2.yaml", moved)
    relocated = dict(spec, models={**spec["models"], "asm1_bsm2": {"source": str(moved)}})
    assert spec_fingerprint(relocated) == base
    # Changing what is simulated changes it.
    tank = {**spec["units"]["tank1"], "volume": 1600.0}
    assert (
        spec_fingerprint(dict(spec, units={**spec["units"], "tank1": tank}), base_dir=path.parent)
        != base
    )
    assert (
        spec_fingerprint(dict(spec, connections=spec["connections"][:-1]), base_dir=path.parent)
        != base
    )
    model = tmp_path / "bsm2" / "models" / "asm1_bsm2.yaml"
    model.write_text(model.read_text().replace("muH: {value: 4.0", "muH: {value: 4.5"))
    unpinned = dict(
        spec, models={**spec["models"], "asm1_bsm2": {"source": "models/asm1_bsm2.yaml"}}
    )
    assert spec_fingerprint(unpinned, base_dir=path.parent) != base
    # A plant written again from its rebuilt self has the same fingerprint.
    rebuilt = Plant.from_spec(BSM2_PROJECT / "flowsheet.yaml").to_spec(tmp_path / "again.yaml")
    assert spec_fingerprint(rebuilt, base_dir=tmp_path) == spec_fingerprint(
        BSM2_PROJECT / "flowsheet.yaml"
    )


def test_the_fingerprint_reads_current_content_not_recorded_hashes(tmp_path):
    """An edited model changes the fingerprint even though the spec still records the
    old hash; spec_changes names the edit."""
    from aquakin.plant.spec import spec_changes, spec_fingerprint, spec_inputs

    shutil.copytree(BSM2_PROJECT, tmp_path / "bsm2")
    path = tmp_path / "bsm2" / "flowsheet.yaml"
    before = spec_inputs(path)
    assert spec_changes(path) == []
    model = tmp_path / "bsm2" / "models" / "asm1_bsm2.yaml"
    model.write_text(model.read_text().replace("muH: {value: 4.0", "muH: {value: 4.5"))
    after = spec_inputs(path)
    assert after["model:asm1_bsm2"] != before["model:asm1_bsm2"]
    assert after["flowsheet"] == before["flowsheet"] and after["model:adm1"] == before["model:adm1"]
    assert spec_fingerprint(path) != spec_fingerprint(BSM2_PROJECT / "flowsheet.yaml")
    assert spec_changes(path) == ["model 'asm1_bsm2' (models/asm1_bsm2.yaml) has changed"]


def test_the_fingerprint_follows_order_but_not_spelling():
    from aquakin.plant.spec import spec_fingerprint

    spec = _bsm1().to_spec()
    base = spec_fingerprint(spec)
    spelled = [c.replace(" -> ", "->") if isinstance(c, str) else c for c in spec["connections"]]
    as_maps = [
        {"from": c.split("->")[0].strip(), "to": c.split("->")[1].strip()}
        if isinstance(c, str)
        else c
        for c in spec["connections"]
    ]
    bare = [
        c.replace("tank1.out", "tank1") if isinstance(c, str) else c for c in spec["connections"]
    ]
    for connections in (spelled, as_maps, bare):
        assert spec_fingerprint(dict(spec, connections=connections)) == base
    reordered = dict(spec, units=dict(reversed(list(spec["units"].items()))))
    assert spec_fingerprint(reordered) != base


def test_a_model_file_edited_after_loading_is_refused(tmp_path):
    shutil.copytree(BSM2_PROJECT, tmp_path / "bsm2")
    p = Plant.from_spec(tmp_path / "bsm2" / "flowsheet.yaml")
    model = tmp_path / "bsm2" / "models" / "asm1_bsm2.yaml"
    model.write_text(model.read_text().replace("muH: {value: 4.0", "muH: {value: 7.0"))
    with pytest.raises(SpecError, match="has changed since model 'asm1_bsm2' was loaded"):
        p.to_spec(tmp_path / "again.yaml")


def test_files_from_outside_the_project_are_copied_in_and_never_overwritten(tmp_path):
    from aquakin.plant import read_influent_table, write_influent_table

    outside = write_influent_table(
        load_bsm1_influent("dry", _asm1()), tmp_path / "elsewhere" / "dry.csv"
    )
    p = build_bsm1()
    p.add_influent("feed", read_influent_table(outside, _asm1()), to=p.influent_endpoint)
    project = tmp_path / "project"
    spec = p.to_spec(project / "flowsheet.yaml")
    assert spec["units"]["feed"]["series"] == "data/feed.csv"
    assert (project / "data" / "feed.csv").read_bytes() == outside.read_bytes()
    assert not any(str(tmp_path / "elsewhere") in str(v) for v in spec["models"].values())
    # A different table already at data/feed.csv is not overwritten.
    (project / "data" / "feed.csv").write_text("t,Q\n0,1\n1,1\n")
    with pytest.raises(SpecError, match="already exists with different content"):
        p.to_spec(project / "flowsheet.yaml")


def test_yaml_1_2_reads_leading_zeros_and_dates_as_written(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("a: 010\nb: 2026-01-01\nc: 0x1F\nd: on\n")
    assert load_spec_file(path) == {"a": 10, "b": "2026-01-01", "c": 31, "d": "on"}


def test_nan_and_misspelled_settings_are_refused_before_anything_is_written(tmp_path):
    from aquakin.plant.spec import write_spec

    target = tmp_path / "flowsheet.yaml"
    target.write_text("kept: true\n")
    spec = _bsm1().to_spec()
    for change, message in (
        ({"volume": float("nan")}, "'volume' must be a real number, got nan"),
        ({"volumee": 1000.0}, "'volumee' is not a setting of cstr"),
    ):
        bad = dict(spec, units={**spec["units"], "tank1": {**spec["units"]["tank1"], **change}})
        with pytest.raises(SpecError, match=message):
            write_spec(bad, target)
        assert target.read_text() == "kept: true\n"
    with pytest.raises(SpecError, match="units.tank1"):
        Plant.from_spec(
            dict(
                spec,
                units={
                    **spec["units"],
                    "tank1": {**spec["units"]["tank1"], "volume": float("nan")},
                },
            )
        )


def test_a_draft_with_a_required_setting_unset_can_be_saved(tmp_path):
    from aquakin.plant.spec import check_spec, write_spec

    spec = _bsm1().to_spec()
    draft = dict(spec, units={**spec["units"], "tank1": {**spec["units"]["tank1"], "volume": None}})
    written = write_spec(draft, tmp_path / "draft.yaml")
    assert written["units"]["tank1"]["volume"] is None
    assert not check_spec(tmp_path / "draft.yaml")["ok"]


def test_whole_and_decimal_numbers_are_one_plant():
    from aquakin.plant.spec import spec_fingerprint

    spec = _bsm1().to_spec()
    tank = spec["units"]["tank1"]
    as_int = dict(
        spec,
        units={
            **spec["units"],
            "tank1": {
                **tank,
                "volume": int(tank["volume"]),
                "conditions": {
                    k: int(v) if float(v).is_integer() else v for k, v in tank["conditions"].items()
                },
            },
        },
    )
    assert spec_fingerprint(as_int) == spec_fingerprint(spec)


def _chain(folder: Path, top_extends: str) -> Path:
    """A model three files deep: mine -> base1 -> base0 -> shipped asm1."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "base0.yaml").write_text(
        "model:\n  name: base0\n  extends: asm1  # the shipped model\nparameters:\n  muH: {value: 5.0}\n"
    )
    (folder / "base1.yaml").write_text(
        "model:\n  name: base1\n  extends: base0.yaml\nparameters:\n  KS: {value: 12.0}\n"
    )
    top = folder / "mine.yaml"
    top.write_text(
        f"# a variant with a comment\nmodel:\n  name: mine\n  extends: {top_extends}  # parent\n"
        "parameters:\n  bH: {value: 0.5}\n"
    )
    return top


def _plant_of(model):
    p = Plant("p")
    p.add_unit(CSTRUnit("tank", model, 100.0, ["in"], conditions={"T": 293.15}))
    p.add_influent("feed", InfluentSeries.constant(model, Q=10.0, **FEED), to="tank.in")
    return p


def test_a_whole_extends_chain_from_outside_is_copied_in(tmp_path):
    from aquakin.plant.spec import spec_fingerprint

    top = _chain(tmp_path / "elsewhere", "base1.yaml")
    model = aquakin.load_model_from_file(top)
    p = _plant_of(model)
    project = tmp_path / "project"
    spec = p.to_spec(project / "flowsheet.yaml")
    assert spec["models"]["mine"]["source"] == "models/mine.yaml"
    copied = (project / "models" / "mine.yaml").read_text()
    assert "extends: base1.yaml  # parent" in copied and "# a variant with a comment" in copied
    assert "extends: base0.yaml" in (project / "models" / "base1.yaml").read_text()
    assert (project / "models" / "base0.yaml").is_file()
    # The copies build the same model, so the fingerprint is the original's.
    q = Plant.from_spec(project / "flowsheet.yaml")
    np.testing.assert_array_equal(
        np.asarray(q.default_parameters()), np.asarray(p.default_parameters())
    )
    assert spec_fingerprint(project / "flowsheet.yaml") == spec_fingerprint(
        p.to_spec(), base_dir=tmp_path
    )


def test_a_project_model_extending_a_file_outside_is_made_self_contained(tmp_path):
    shared = tmp_path / "shared"
    _chain(shared, "base1.yaml")
    project = tmp_path / "project"
    (project / "models").mkdir(parents=True)
    (project / "models" / "mine.yaml").write_text(
        "model:\n  name: mine\n  extends: ../../shared/base1.yaml\nparameters:\n  bH: {value: 0.5}\n"
    )
    p = _plant_of(aquakin.load_model_from_file(project / "models" / "mine.yaml"))
    p.to_spec(project / "flowsheet.yaml")
    assert "extends: base1.yaml" in (project / "models" / "mine.yaml").read_text()
    shutil.rmtree(shared)  # the project no longer needs it
    Plant.from_spec(project / "flowsheet.yaml")


def test_a_broken_chain_is_refused_and_nothing_is_written(tmp_path):
    top = _chain(tmp_path / "elsewhere", "base1.yaml")
    model = aquakin.load_model_from_file(top)
    p = _plant_of(model)
    spec = p.to_spec()
    (tmp_path / "elsewhere" / "base0.yaml").unlink()
    project = tmp_path / "project"
    with pytest.raises(SpecError, match="base0.yaml', which does not exist"):
        aquakin.plant.write_spec(spec, project / "flowsheet.yaml", base_dir=tmp_path)
    assert not project.exists()


def test_an_in_memory_table_can_be_saved_twice(tmp_path):
    p = _dynamic_bsm1()
    first = p.to_spec(tmp_path / "flowsheet.yaml")
    assert p.to_spec(tmp_path / "flowsheet.yaml")["files"] == first["files"]


def test_a_file_model_may_not_take_a_shipped_name(tmp_path):
    path = tmp_path / "asm1.yaml"
    path.write_text("model:\n  name: asm1\n  extends: asm1\nparameters:\n  muH: {value: 5.0}\n")
    p = _plant_of(aquakin.load_model_from_file(path))
    with pytest.raises(SpecError, match="has the name of a shipped model"):
        p.to_spec()
    spec = {"format": 2, "models": {"asm1": {"source": str(path)}}, "units": {}}
    with pytest.raises(SpecError, match="'asm1' is the name of a shipped model"):
        Plant.from_spec(spec)


def test_an_unreadable_spec_file_is_a_spec_error(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("units: [unclosed\n")
    with pytest.raises(SpecError, match="not a readable spec file"):
        Plant.from_spec(path)


def test_a_stream_may_not_share_an_influent_name():
    from aquakin.plant.errors import WiringError

    p = _bsm1()
    with pytest.raises(WiringError, match="is the name of an influent"):
        p.register_stream("feed", "clarifier.overflow")
    q = build_bsm1()
    q.register_stream("raw", "clarifier.overflow")
    with pytest.raises(WiringError, match="is already the name of a stream"):
        q.add_influent("raw", InfluentSeries.constant(_asm1(), Q=1.0), to=q.influent_endpoint)
