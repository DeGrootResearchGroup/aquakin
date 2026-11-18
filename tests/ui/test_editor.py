"""The editor page, built and driven without a browser."""

import pytest
from trame.app import get_server

from aquakin_ui.app import Editor

from .conftest import PROJECTS


@pytest.fixture(scope="module")
def editor():
    server = get_server(name="aquakin-ui-test", client_type="vue3")
    ed = Editor(server, path=PROJECTS / "bsm1" / "flowsheet.yaml")
    yield ed
    ed.worker.stop()


def test_the_page_draws_the_flowsheet(editor):
    s = editor.state
    assert len(s.nodes) == 10 and len(s.edges) == 11
    assert {n["id"] for n in s.nodes} >= {"influent", "tank1", "clarifier"}
    tank3 = next(n for n in s.nodes if n["id"] == "tank3")
    assert tank3["data"]["label"] == "Aerobic zone 1" and tank3["data"]["icon"] == "cstr-aerated"
    assert sorted(s.products) == ["clarifier.overflow", "underflow_split.waste"]


def test_checks_come_from_aquakin(editor):
    for uid in editor.doc.units:
        assert editor._check_unit(uid) is None
    assert editor._check_plant()["ok"]
    editor.publish(checks=False)
    assert editor.state.errors == [] and editor.state.n_recycles == 2


def test_a_dropped_unit_is_built_by_aquakin_and_its_error_shown(editor):
    editor.on_dropped({"kind": "cstr", "position": {"x": 0, "y": 600}})
    uid = editor.state.selected
    assert editor._check_unit(uid) == "ValueError: 'volume' is required"
    editor.publish(checks=False)
    assert f"{uid}: ValueError: 'volume' is required" in editor.state.errors
    editor.on_remove({"nodes": [uid], "edges": []})
    assert uid not in editor.doc.units


def test_removing_a_connected_port_asks_first(editor):
    from aquakin_ui import forms

    editor.on_select({"nodes": ["inlet_mix"], "edges": []})
    rows = forms.rows_for(editor.doc, "inlet_mix", editor.model_info)
    for r in rows:
        if r["key"] == "input_port_names":
            r["value"] = "fresh, ras"
    before = list(editor.doc.connections)
    editor.apply_properties(rows)
    assert editor.state.confirm_open
    assert editor.state.confirm_lines == [
        "tank5_split.internal_recycle -> inlet_mix.internal_recycle"
    ]
    assert editor.doc.connections == before  # nothing changed yet
    editor.cancel_apply()
    assert not editor.state.confirm_open and editor.doc.connections == before
    editor.apply_properties(rows)
    editor.confirm_apply()
    assert not editor.state.confirm_open and len(editor.doc.connections) == len(before) - 1
    assert editor.doc.ports("inlet_mix")[0] == ["fresh", "ras"]


def test_save_writes_through_aquakin_and_open_reads_it_back(tmp_path):
    """Save goes through aquakin's spec writer (an outside model file is copied
    into the project, the aquakin version recorded) and Open reads YAML 1.2."""
    import shutil

    project = tmp_path / "bsm2"
    shutil.copytree(PROJECTS / "bsm2", project)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (project / "models" / "asm1_bsm2.yaml").rename(outside / "asm1_bsm2.yaml")
    spec = (project / "flowsheet.yaml").read_text()
    (project / "flowsheet.yaml").write_text(
        spec.replace("models/asm1_bsm2.yaml", str(outside / "asm1_bsm2.yaml"))
    )
    server = get_server(name="aquakin-ui-save-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        n_units = len(ed.doc.units)
        ed.save()
        assert (project / "models" / "asm1_bsm2.yaml").read_text() == (
            outside / "asm1_bsm2.yaml"
        ).read_text()
        text = (project / "flowsheet.yaml").read_text()
        assert "models/asm1_bsm2.yaml" in text and "aquakin:" in text
        assert str(outside) not in text
        ed.open(project / "flowsheet.yaml")
        assert len(ed.doc.units) == n_units
    finally:
        ed.worker.stop()


def _study_project(tmp_path):
    """A one-tank project with one simulate study, written with aquakin."""
    import aquakin
    from aquakin.plant import CSTRUnit, InfluentSeries, Plant

    m = aquakin.load_model("asm1")
    p = Plant("tank")
    p.add_unit(CSTRUnit("tank", m, 1000.0, ["in"], conditions={"T": 293.15}))
    p.add_influent("feed", InfluentSeries.constant(m, Q=2000.0, SS=60.0, SNH=30.0), to="tank.in")
    project = tmp_path / "project"
    p.to_spec(project / "flowsheet.yaml")
    study = project / "studies" / "short"
    study.mkdir(parents=True)
    (study / "study.yaml").write_text(
        "study:\n  kind: simulate\n  description: half a day\n"
        "  time: {t_span: [0, 0.5], t_eval: {start: 0, stop: 0.5, num: 3}}\n"
        "  streams: [tank]\n"
    )
    return project


def test_studies_are_listed_run_frozen_and_copied(tmp_path):
    project = _study_project(tmp_path)
    server = get_server(name="aquakin-ui-studies-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        [study] = ed.state.studies
        assert study["name"] == "short" and study["state"] == "not run"
        assert study["kind"] == "simulate" and study["description"] == "half a day"

        events = []
        original = ed._apply_run_event
        ed._apply_run_event = lambda name, event: (events.append(event), original(name, event))
        ed.run_study("short")
        assert ed._runs["short"].wait(300)["event"] == "done"
        assert events[-1] == {"event": "done", "state": "fresh", "reasons": [], "notes": []}
        assert ed._study("short")["state"] == "fresh" and not ed._runs

        # Running again would replace the results: the editor asks, and "copy"
        # keeps a copy before running.
        ed.run_study("short")
        assert ed.state.study_confirm_open and "Replace the results" in ed.state.study_confirm_title
        ed.answer_study("copy")
        assert "short" in ed._runs, ed.state.message
        run = ed._runs["short"]
        assert run.wait(300)["event"] == "done"
        assert {st["name"] for st in ed.state.studies} == {"short", "short_copy"}

        ed.freeze_study("short")
        assert ed._study("short")["frozen"]
        ed.unfreeze_study("short")
        assert "delete its results" in ed.state.study_confirm_title
        ed.answer_study("cancel")
        assert ed._study("short")["frozen"]
        ed.unfreeze_study("short")
        ed.answer_study("go")
        assert not ed._study("short")["frozen"] and ed._study("short")["state"] == "not run"
        assert ed._study("short_copy")["state"] == "fresh"
    finally:
        ed.stop_runs()
        ed.worker.stop()


def test_a_run_can_be_cancelled(tmp_path):
    project = _study_project(tmp_path)
    (project / "studies" / "short" / "study.yaml").write_text(
        "study:\n  kind: simulate\n  time: {t_span: [0, 50]}\n  streams: [tank]\n"
        "  initial_state: steady\n"
    )
    server = get_server(name="aquakin-ui-cancel-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        ed.run_study("short")
        run = ed._runs["short"]
        ed.cancel_run("short")
        assert run.wait(300)["event"] == "cancelled"
        assert ed._study("short")["state"] == "cancelled"
    finally:
        ed.stop_runs()
        ed.worker.stop()


def test_save_refuses_a_file_changed_on_disk(tmp_path):
    project = _study_project(tmp_path)
    server = get_server(name="aquakin-ui-guard-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        spec = project / "flowsheet.yaml"
        spec.write_text(spec.read_text() + "\n# edited elsewhere\n")
        ed.save()
        assert "has changed on disk" in ed.state.message
        assert spec.read_text().endswith("# edited elsewhere\n")
        ed.state.spec_path = str(project / "other.yaml")
        ed.save()
        assert (project / "other.yaml").exists()
        ed.state.spec_path = str(spec)
        ed.save()
        assert "already exists" in ed.state.message
    finally:
        ed.worker.stop()
