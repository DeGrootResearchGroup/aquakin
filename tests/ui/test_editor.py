"""The editor page, built and driven without a browser."""

import pytest
import yaml
from trame.app import get_server

from aquakin_ui.app import Editor
from aquakin_ui.document import edge_id
from aquakin_ui.settings import Settings
from aquakin_ui.worker import AquakinWorker, WorkerError

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
    assert editor._check_unit(uid) == "'volume' is required"
    editor.publish(checks=False)
    assert f"{uid}: 'volume' is required" in editor.state.errors
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
        assert events[-1] == {
            "event": "done",
            "state": "fresh",
            "has_results": True,
            "reasons": [],
            "notes": [],
        }
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


def test_simulate_saves_sets_up_runs_and_shows_the_plant(tmp_path):
    project = _study_project(tmp_path)
    server = get_server(name="aquakin-ui-simulate-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        # A moved unit is an unsaved change: Simulate saves it first.
        ed.on_move([{"id": "tank", "position": {"x": 300.0, "y": 40.0}}])
        before = (project / "flowsheet.yaml").read_text()
        ed.simulate()
        assert (project / "flowsheet.yaml").read_text() != before
        study = yaml.safe_load((project / "studies" / "simulation" / "study.yaml").read_text())
        assert study["study"]["streams"] == ["tank.out"]
        assert study["study"]["observables"][0]["channels"][:3] == ["Q", "COD", "TKN"]
        assert ed.state.panel == "studies" and ed.state.simulating
        assert ed._runs["simulation"].wait(300)["event"] == "done"
        assert not ed.state.simulating
        # The results open when the run finishes.
        assert ed.state.results_open and ed.state.results_mode == "steady"
        table = ed.state.results_measured
        assert table["columns"] == ["tank.out"]
        assert table["rows"][0] == {"channel": "Q", "unit": "m³/d", "values": ["2,000"]}

        # Again: nothing to save, and replacing the results asks first, unless
        # the user said not to ask again (kept as their preference).
        ed.state.results_open = False
        ed.simulate()
        assert ed.state.study_confirm_open and "Replace the results" in ed.state.study_confirm_title
        assert ed.state.study_confirm_kind == "run" and not ed.state.study_confirm_dont_ask
        ed.state.study_confirm_dont_ask = True
        ed.answer_study("cancel")  # not gone ahead: nothing is kept
        assert ed.state.ask_replace and ed.settings["ask_before_replacing"]
        ed.simulate()
        ed.state.study_confirm_dont_ask = True
        ed.answer_study("go")
        assert not ed.state.ask_replace and not Settings()["ask_before_replacing"]
        assert ed._runs["simulation"].wait(300)["event"] == "done"
        ed.simulate()  # no question now
        assert not ed.state.study_confirm_open and "simulation" in ed._runs
        assert ed._runs["simulation"].wait(300)["event"] == "done"
        ed.set_ask_replace(True)  # the Studies tab's switch
        assert ed.state.ask_replace and Settings()["ask_before_replacing"]

        # A run in time shows a chart per channel; another kind lists its files.
        ed.run_study("short")
        assert ed._runs["short"].wait(300)["event"] == "done"
        ed.show_results("short")
        assert ed.state.results_mode == "time" and ed.state.results_stream == "tank"
        assert [c["channel"] for c in ed.state.results_charts] == ["Q", "SI", "SS", "XI"]
        ed.show_charts(["SNH"])
        assert "<polyline" in ed.state.results_charts[0]["svg"]
    finally:
        ed.stop_runs()
        ed.worker.stop()


class _Failing:
    """The worker, failing the commands named: ``fail`` as aquakin would (a
    ``WorkerError``), ``crash`` with any other error."""

    def __init__(self):
        self.worker = AquakinWorker()
        self.fail: set[str] = set()
        self.crash: set[str] = set()

    def request(self, command, **fields):
        if command in self.fail:
            raise WorkerError(f"{command} failed", "StudyError")
        if command in self.crash:
            raise RuntimeError(f"{command} crashed")
        return self.worker.request(command, **fields)

    def stop(self):
        self.worker.stop()


def test_simulate_results_and_the_dialog_say_what_went_wrong(tmp_path):
    project = _study_project(tmp_path)
    (project / "flowsheet.yaml").rename(project / "plant.yaml")
    blocked = tmp_path / "blocked"
    blocked.write_text("")  # a file where the preferences folder would be
    worker = _Failing()
    server = get_server(name="aquakin-ui-failing-test", client_type="vue3")
    ed = Editor(
        server, worker=worker, path=project / "plant.yaml", settings=Settings(blocked / "ui.json")
    )
    try:
        worker.fail = {"simulation_study"}
        ed.simulate()
        assert ed.state.message == "could not set up the simulation: simulation_study failed"
        worker.fail = set()
        ed.simulate()
        study = yaml.safe_load(
            (project / "studies" / "simulation_plant" / "study.yaml").read_text()
        )
        assert study["study"]["flowsheet"] == "../../plant.yaml"  # not flowsheet.yaml
        ed.state.panel = "properties"
        ed.simulate()  # already running: shows it
        assert ed.state.panel == "studies" and not ed.state.study_confirm_open
        assert ed._runs["simulation_plant"].wait(300)["event"] == "done"

        ed.show_results("nope")
        ed.show_results("short")
        assert "could not read the results of short" in ed.state.message
        assert "has no results; run it first" in ed.state.message
        # Results gone stale say why, as aquakin's check of the study does.
        settings = project / "studies" / "simulation_plant" / "study.yaml"
        edited = yaml.safe_load(settings.read_text())
        edited["study"]["description"] = "edited after the run"
        settings.write_text(yaml.safe_dump(edited))
        ed.show_results("simulation_plant")
        assert ed.state.results_note == (
            "Run the study again for up-to-date results: the study's settings have changed."
        )

        ed.set_ask_replace(False)
        assert not ed.state.ask_replace and "could not save the preference" in ed.state.message
        ed.state.message = ""
        ed.set_ask_replace(False)  # no change: nothing to keep, nothing said
        assert ed.state.message == ""

        # A Fisher study's results are shown as tables.
        fisher = project / "studies" / "design"
        (fisher / "results").mkdir(parents=True)
        (fisher / "study.yaml").write_text(
            "study: {kind: fisher, free: [asm1.muH], observables: [{stream: tank}], "
            "t_obs: [1.0], sigma: 1.0}\n"
        )
        (fisher / "results" / "fisher.csv").write_text(",asm1.muH\nasm1.muH,4.0\n")
        (fisher / "results" / "summary.yaml").write_text(
            "values: {asm1.muH: 6.0}\nstd: {asm1.muH: 0.5}\ncriteria: {D: 1.386}\n"
        )
        ed.state.studies = [
            *ed.state.studies,
            {"name": "design", "path": str(fisher), "state": "fresh"},
        ]
        ed.show_results("design")
        assert ed.state.results_mode == "analysis"
        assert ed.state.results_files == ["fisher.csv", "summary.yaml"]
        assert ed.state.results_tables[0]["rows"] == [["asm1.muH", "6.000", "0.5000"]]

        worker.fail = {"study_form"}
        ed.new_study()
        assert ed.state.message.startswith("could not read the study settings")
        worker.fail = {"plant_choices"}  # the plant does not build: the drawn ports
        ed.new_study()
        assert ed.state.study_dialog_open and "tank.out" in ed.state.study_choices["streams"]
        worker.fail, worker.crash = set(), {"plant_choices"}
        ed.new_study()
        assert ed.state.study_choices == {"streams": [], "parameters": [], "channels": {}}
        ed.edit_study("nope")
        worker.fail = {"read_study"}
        ed.edit_study("short")
        assert ed.state.message.startswith("could not read study short")
    finally:
        ed.stop_runs()
        ed.worker.stop()


def test_studies_follow_the_spec_that_is_open(tmp_path):
    project = _study_project(tmp_path)
    (project / "second.yaml").write_text((project / "flowsheet.yaml").read_text())
    server = get_server(name="aquakin-ui-follow-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")

    def settings(name):
        return yaml.safe_load((project / "studies" / name / "study.yaml").read_text())["study"]

    try:
        ed.set_ask_replace(False)
        ed.simulate()
        assert "flowsheet" not in settings("simulation")
        assert ed._runs["simulation"].wait(300)["event"] == "done"
        # A simulation pointed at another file of the folder is pointed back.
        study = settings("simulation")
        (project / "studies" / "simulation" / "study.yaml").write_text(
            yaml.safe_dump({"study": {**study, "flowsheet": "../../second.yaml"}})
        )
        ed.simulate()
        assert "flowsheet" not in settings("simulation")
        assert ed._runs["simulation"].wait(300)["event"] == "done"

        # Another spec in the folder has its own simulation and its own studies.
        ed.open(project / "second.yaml")
        ed.simulate()
        assert settings("simulation_second")["flowsheet"] == "../../second.yaml"
        assert ed._runs["simulation_second"].wait(300)["event"] == "done"
        ed.freeze_study("simulation_second")
        ed.simulate()
        assert "simulation_second is frozen" in ed.state.message and not ed._runs

        ed.new_study()
        ed.save_study_dialog("fit", "simulate", {**ed.state.study_draft, "streams": ["tank.out"]})
        assert settings("fit")["flowsheet"] == "../../second.yaml"
        ed.edit_study("fit")
        ed.save_study_dialog("fit", "simulate", {**ed.state.study_draft, "description": "kept"})
        assert settings("fit") == {
            "kind": "simulate",
            "flowsheet": "../../second.yaml",
            "description": "kept",
            "streams": ["tank.out"],
        }
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


def test_a_study_is_set_up_and_edited_in_the_dialog(tmp_path):
    project = _study_project(tmp_path)
    server = get_server(name="aquakin-ui-study-dialog-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        ed.new_study()
        s = ed.state
        assert s.study_dialog_open and s.study_kind == "simulate" and not s.study_editing
        assert "tank.out" in s.study_choices["streams"]
        assert "asm1.muH" in s.study_choices["parameters"]
        assert "TN" in s.study_choices["channels"]["tank.out"]

        # Switching kinds keeps what both kinds have.
        draft = dict(s.study_draft, description="Ammonia design")
        ed.set_study_kind("fisher", draft)
        assert s.study_draft["description"] == "Ammonia design"
        names = [f["name"] for f in s.study_fields]
        assert names[:3] == ["description", "parameters", "initial_state"]

        draft = dict(s.study_draft)
        draft["free"] = ["asm1.muH"]
        draft["observables"] = [{"stream": "tank.out", "channels": ["SNH"], "measures": {}}]
        draft["t_obs"] = {"mode": "list", "start": "", "stop": "", "num": "", "list": "0.1, 0.2"}
        draft["sigma"] = "lots"
        ed.save_study_dialog("design", "fisher", draft)
        assert s.study_dialog_open and "Measurement error: 'lots' is not a number" in s.study_error
        ed.save_study_dialog("design", "fisher", {**draft, "sigma": ""})
        assert s.study_error == "Still needed: Measurement error."
        ed.save_study_dialog("", "fisher", {**draft, "sigma": "0.1"})
        assert s.study_error == "Give the study a name."
        ed.save_study_dialog("design", "fisher", {**draft, "sigma": "0.1"})
        assert not s.study_dialog_open
        written = yaml.safe_load((project / "studies" / "design" / "study.yaml").read_text())
        assert written["study"] == {
            "kind": "fisher",
            "description": "Ammonia design",
            "free": ["asm1.muH"],
            "observables": [{"stream": "tank.out", "channels": ["SNH"]}],
            "t_obs": [0.1, 0.2],
            "sigma": 0.1,
        }
        assert {st["name"] for st in s.studies} == {"short", "design"}

        # Editing opens the study's settings; a name already taken is refused.
        ed.edit_study("design")
        assert s.study_editing.endswith("design") and s.study_draft["sigma"] == "0.1"
        ed.save_study_dialog("design", "fisher", {**s.study_draft, "sigma": "0.2"})
        assert (
            yaml.safe_load((project / "studies" / "design" / "study.yaml").read_text())["study"][
                "sigma"
            ]
            == 0.2
        )
        ed.new_study()
        ed.save_study_dialog("design", "simulate", {**s.study_draft, "streams": ["tank.out"]})
        assert s.study_error.startswith("a study named design already exists")
    finally:
        ed.worker.stop()


def _rows_with(ed, key, value):
    rows = [dict(r) for r in ed.state.sel_rows]
    next(r for r in rows if r["key"] == key)["value"] = value
    return rows


def test_edits_can_be_undone_and_unapplied_form_edits_are_not_lost(tmp_path):
    project = _study_project(tmp_path)
    server = get_server(name="aquakin-ui-undo-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        assert not ed.state.can_undo
        ed.on_remove({"nodes": ["tank"], "edges": []})
        assert "tank" not in ed.doc.units and "Undo brings it back" in ed.state.message
        ed.undo()
        assert "tank" in ed.doc.units and ed.doc.connections and not ed.state.can_undo
        ed.undo()  # nothing left: nothing happens
        [wire] = ed.doc.connections
        ed.on_remove({"nodes": [], "edges": [edge_id(wire)]})
        ed.on_connect(
            {"source": wire[0], "sourceHandle": wire[1], "target": wire[2], "targetHandle": wire[3]}
        )
        assert ed.doc.connections == [wire]
        ed.undo()
        assert ed.doc.connections == []
        ed.undo()
        assert ed.doc.connections == [wire] and not ed.state.can_undo
        ed.state.selected = None
        assert ed.apply_properties([]) is False
        ed.answer_study("cancel")  # no question asked: nothing to do

        # A refused wire is no edit; a move and an Apply are.
        ed.on_connect(
            {"source": "tank", "sourceHandle": "out", "target": "tank", "targetHandle": "in"}
        )
        assert not ed.state.can_undo
        ed.on_move([{"id": "tank", "position": {"x": 999, "y": 0}}])
        ed.on_select({"nodes": ["tank"], "edges": []})
        assert not ed.state.sel_dirty
        assert ed.apply_properties(_rows_with(ed, "volume", "1500"))
        ed.undo()
        assert ed.doc.units["tank"].settings["volume"] == 1000.0
        ed.undo()
        assert ed.doc.units["tank"].position["x"] != 999

        # Selecting another unit applies the form's unapplied edits first; edits
        # that cannot be read keep the unit selected (nothing typed is lost).
        ed.on_select({"nodes": ["tank"], "edges": []})
        ed.on_select({"nodes": ["feed"], "edges": []}, _rows_with(ed, "volume", "1200"))
        assert ed.doc.units["tank"].settings["volume"] == 1200.0
        assert ed.state.selected == "feed"
        ed.on_select({"nodes": ["tank"], "edges": []})
        ed.on_select({"nodes": ["feed"], "edges": []}, _rows_with(ed, "volume", "big"))
        assert ed.state.selected == "tank"
        assert ed.state.message == "tank: volume: 'big' is not a number"

        # Cancel in the remove-connections question sends the unit's rows again.
        ed.state.flush()
        ed.on_select({"nodes": ["feed"], "edges": []})
        ed.state.flush()
        before = ed.state.sel_rows
        ed.cancel_apply()
        assert ed.state.sel_rows != before and ed.state.sel_rows[1:] == before[1:]

        # Opening another file loses the history.
        ed.on_move([{"id": "tank", "position": {"x": 5, "y": 5}}])
        ed.state.spec_path = str(project / "flowsheet.yaml")
        ed.request_open()
        assert ed.state.unsaved_open and ed.state.unsaved_can_save
        ed.answer_unsaved("discard")
        assert not ed.state.can_undo and ed.doc.units["tank"].position["x"] != 5
    finally:
        ed.worker.stop()


def test_unsaved_edits_are_saved_or_kept_before_open_and_run(tmp_path):
    project = _study_project(tmp_path)
    (project / "other.yaml").write_text((project / "flowsheet.yaml").read_text())
    server = get_server(name="aquakin-ui-unsaved-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        ed.state.spec_path = str(project / "other.yaml")
        ed.request_open()  # nothing unsaved: opens at once
        assert not ed.state.unsaved_open and ed._opened[0] == project / "other.yaml"

        ed.on_move([{"id": "tank", "position": {"x": 777, "y": 0}}])
        ed.state.spec_path = str(project / "flowsheet.yaml")
        ed.request_open()
        assert ed.state.unsaved_open and "leave the unsaved edits" in ed.state.unsaved_text
        ed.answer_unsaved("cancel")
        assert ed._opened[0] == project / "other.yaml"
        ed.request_open()
        ed.answer_unsaved("save")  # saved to other.yaml, then flowsheet.yaml opened
        assert "777" in (project / "other.yaml").read_text()
        assert ed._opened[0] == project / "flowsheet.yaml"

        # Run brings the file up to the drawing before a live study runs it.
        ed.refresh_studies()
        ed.on_move([{"id": "tank", "position": {"x": 555, "y": 0}}])
        ed.run_study("short")
        assert "555" in (project / "flowsheet.yaml").read_text()
        assert ed._runs["short"].wait(300)["event"] == "done"

        # A replace question answered Cancel does not leave results to pop up later.
        ed._show_when_done.add("short")
        ed.run_study("short")
        ed.answer_study("cancel")
        assert "short" not in ed._show_when_done
        # Results just run are not called stale by an older listing: where the
        # study stands comes with its results.
        ed.state.studies = [{**st, "state": "stale"} for st in ed.state.studies]
        ed.show_results("short")
        assert ed.state.results_note == ""
        ed.state.spec_path = "  "
        ed.request_open()
        assert ed.state.message == "type the path of a flowsheet file to open"
    finally:
        ed.stop_runs()
        ed.worker.stop()


def test_a_flowsheet_without_a_file_asks_where_to_keep_it(tmp_path):
    server = get_server(name="aquakin-ui-nofile-test", client_type="vue3")
    ed = Editor(server)
    try:
        assert ed.state.spec_path == "" and not ed._unsaved()
        ed.refresh_studies()
        assert ed.state.studies == []
        ed.on_dropped({"kind": "cstr", "position": {"x": 0, "y": 0}})
        assert ed._unsaved()
        ed.simulate()
        assert "the flowsheet has no file yet" in ed.state.message
        ed.save()
        assert "type where to keep the flowsheet" in ed.state.message
        ed.new_study()
        assert "save the flowsheet first" in ed.state.message and not ed.state.study_dialog_open
        ed.request_open()  # nothing typed: nothing to open, and nothing asked
        assert not ed.state.unsaved_open
        ed.state.spec_path = str(tmp_path / "new" / "plant.yaml")
        ed.request_open()  # unsaved edits and no file of their own: only Discard
        assert ed.state.unsaved_open and not ed.state.unsaved_can_save
        ed.answer_unsaved("cancel")
    finally:
        ed.worker.stop()


def test_toolbar_actions_apply_the_forms_edits_and_aeration_suits_the_model(tmp_path):
    project = _study_project(tmp_path)
    server = get_server(name="aquakin-ui-pending-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        ed.on_select({"nodes": ["tank"], "edges": []})
        ed.save_button(_rows_with(ed, "volume", "1300"))
        assert "volume: 1300.0" in (project / "flowsheet.yaml").read_text()
        ed.save_button(_rows_with(ed, "volume", "lots"))  # not applied, not saved
        assert "volume: 'lots'" in ed.state.message
        ed.request_open(_rows_with(ed, "volume", "lots"))
        assert not ed.state.unsaved_open
        ed.simulate(_rows_with(ed, "volume", "lots"))
        assert not ed._runs
        ed.run_study("short", _rows_with(ed, "volume", "1350"))
        assert ed.doc.units["tank"].settings["volume"] == 1350.0
        assert "volume: 1350.0" in (project / "flowsheet.yaml").read_text()  # saved first
        assert ed._runs["short"].wait(300)["event"] == "done"

        # Switching on aeration in an ASM3 tank aerates ASM3's oxygen at once.
        rows = ed.state.sel_rows
        ed.reshape_form(rows, next(i for i, r in enumerate(rows) if r["key"] == "model"), "asm3")
        rows = ed.state.sel_rows
        i = next(i for i, r in enumerate(rows) if r["key"] == "aeration.mode")
        ed.reshape_form(rows, i, "kla")
        assert ed.doc.units["tank"].settings["aeration"]["species"] == "SO2"
    finally:
        ed.stop_runs()
        ed.worker.stop()


def test_the_form_marks_the_bad_field_and_new_units_suit_the_model(tmp_path):
    project = _study_project(tmp_path)
    server = get_server(name="aquakin-ui-usability-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        ed.state.panel = "studies"
        ed.on_select({"nodes": ["tank"], "edges": []})
        assert ed.state.panel == "properties"  # its form, from whichever tab was open
        assert not ed.apply_properties(_rows_with(ed, "volume", "big"))
        assert ed.state.sel_errors == {"volume": "'big' is not a number"}
        assert ed.apply_properties(_rows_with(ed, "volume", "900"))
        assert ed.state.sel_errors == {}

        # A new MBR settles the model's particulates, not none.
        ed.on_dropped({"kind": "mbr", "position": {"x": 0, "y": 300}})
        settling = ed.doc.units[ed.state.selected].settings["particulate_species"]
        assert settling and all(s.startswith("X") for s in settling)

        kinds = {k["kind"]: k for group in ed.state.palette for k in group["kinds"]}
        assert kinds["sbr"]["python"] and "only be set from Python" in kinds["sbr"]["summary"]
        assert not kinds["cstr"]["python"]
    finally:
        ed.worker.stop()


def test_an_influent_pattern_is_typed_into_the_form_and_saved_as_text(tmp_path):
    """A feed that follows a schedule is written in the unit form as an
    expression of time; it reaches the spec as the text that was typed, with no
    table file written beside it."""
    import yaml

    project = _study_project(tmp_path)
    server = get_server(name="aquakin-ui-influent-expression-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        ed.on_select({"nodes": ["feed"], "edges": []}, None)
        rows = {r["key"]: r for r in ed.state.sel_rows}
        # The flow is an ordinary editable field, not one the form refuses.
        assert rows["Q"]["widget"] != "readonly"
        assert rows["Q"]["value"] == "2000"

        edited = _rows_with(ed, "Q", "2000 + 500*sin(2*pi*t)")
        next(r for r in edited if r["key"] == "composition")["value"] = (
            "SS: 60, SNH: 30 + 10*sin(2*pi*t)"
        )
        assert ed.apply_properties(edited)

        ed.save()
        doc = yaml.safe_load((project / "flowsheet.yaml").read_text())
        feed = doc["units"]["feed"]
        assert feed["Q"] == "2000 + 500*sin(2*pi*t)"
        assert feed["composition"]["SNH"] == "30 + 10*sin(2*pi*t)"
        assert feed["composition"]["SS"] == 60.0  # a constant stays a number
        assert not (project / "data").exists()

        # And aquakin builds the pattern back from what the editor wrote.
        from aquakin.plant import InfluentExpression, Plant

        rebuilt = Plant.from_spec(project / "flowsheet.yaml")
        feed_back = rebuilt.influents["feed"]
        assert isinstance(feed_back, InfluentExpression)
        assert float(feed_back.at(0.25).Q) == 2500.0
    finally:
        ed.worker.stop()


def test_an_unreadable_influent_pattern_is_refused_by_the_form(tmp_path):
    """aquakin_ui cannot read an expression itself, so the refusal comes from
    aquakin -- but it still has to reach the field rather than the first solve."""
    project = _study_project(tmp_path)
    server = get_server(name="aquakin-ui-influent-expression-bad-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        ed.on_select({"nodes": ["feed"], "edges": []}, None)
        ed.apply_properties(_rows_with(ed, "Q", "2000 + wobble(t)"))
        problem = ed._check_unit("feed")
        assert problem is not None
        assert "wobble" in problem and "'Q'" in problem
    finally:
        ed.worker.stop()


def test_a_channel_of_your_own_is_written_from_the_study_dialog(tmp_path):
    """The channels combobox takes free text, so a measure expression reaches
    aquakin as a plain channel; an unreadable one is refused with its reason
    beside the buttons, before anything is written."""
    project = _study_project(tmp_path)
    server = get_server(name="aquakin-ui-measure-expression-test", client_type="vue3")
    ed = Editor(server, path=project / "flowsheet.yaml")
    try:
        ed.new_study()
        s = ed.state
        draft = dict(s.study_draft)
        draft["streams"] = ["tank.out"]
        draft["observables"] = [
            {
                "stream": "tank.out",
                "channels": ["SNH", "BOD5 = 0.65*[SS] + 0.65*[XS]"],
                "measures": {},
            }
        ]

        # An expression that cannot be read never reaches the file.
        bad = [{**draft["observables"][0], "channels": ["x = [SS]*[XS]"]}]
        ed.save_study_dialog("custom", "simulate", {**draft, "observables": bad})
        assert s.study_dialog_open
        assert "cannot multiply two measured quantities" in s.study_error
        assert not (project / "studies" / "custom").exists()

        ed.save_study_dialog("custom", "simulate", draft)
        assert not s.study_dialog_open
        written = yaml.safe_load((project / "studies" / "custom" / "study.yaml").read_text())
        assert written["study"]["observables"] == [
            {"stream": "tank.out", "channels": ["SNH", "BOD5 = 0.65*[SS] + 0.65*[XS]"]}
        ]

        # It reopens as typed (the dialog shows the expression, not a resolved
        # measure), on a dialog cleared of the earlier refusal.
        ed.edit_study("custom")
        assert not s.study_error
        assert s.study_draft["observables"][0]["channels"] == [
            "SNH",
            "BOD5 = 0.65*[SS] + 0.65*[XS]",
        ]
    finally:
        ed.worker.stop()


def test_errors_reach_the_page_without_class_names_and_simulate_reports_a_broken_study(
    tmp_path,
):
    """The worker sends an error's message and its class apart, so what the page
    shows is the message alone; a Simulate study whose study.yaml cannot be read
    is reported with the reason aquakin gives, and a frozen study without
    results is unfrozen without asking (there is nothing to delete)."""
    project = _study_project(tmp_path)
    worker = _Failing()
    worker.fail = {"check_spec", "check_unit"}
    server = get_server(name="aquakin-ui-plain-errors-test", client_type="vue3")
    ed = Editor(server, worker=worker, path=project / "flowsheet.yaml")
    try:
        assert ed._check_unit("tank") == "check_unit failed"
        assert ed._check_plant() == {"ok": False, "error": "check_spec failed"}
        ed.publish(checks=False)
        assert ed.state.errors == ["tank: check_unit failed"]  # a unit's error first
        worker.fail = set()
        ed._unit_checks.clear()
        assert ed._check_unit("tank") is None
        ed.publish(checks=False)
        assert ed.state.errors == ["plant: check_spec failed"]

        missing = tmp_path / "nowhere.yaml"
        ed.state.spec_path = str(missing)
        ed.request_open()
        assert ed.state.message.startswith(f"could not open {missing}: ")
        assert "No such file" in ed.state.message and "Error" not in ed.state.message

        ed.refresh_studies()
        ed.freeze_study("short")
        assert ed._study("short") == {**ed._study("short"), "frozen": True, "has_results": False}
        ed.unfreeze_study("short")
        assert not ed.state.study_confirm_open and not ed._study("short")["frozen"]

        broken = project / "studies" / "simulation"
        broken.mkdir()
        (broken / "study.yaml").write_text("study: {kind: survey}\n")
        ed.simulate()
        assert ed.state.message.startswith("could not set up the simulation: ")
        assert "kind: Input should be 'simulate'" in ed.state.message
        assert "ValidationError" not in ed.state.message and "simulation" not in ed._runs
    finally:
        ed.stop_runs()
        ed.worker.stop()
