"""The editor page, built and driven without a browser."""

import pytest
import yaml
from trame.app import get_server

from aquakin_ui.app import Editor
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
            raise WorkerError(f"StudyError: {command} failed")
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
        ed.state.studies = [
            {**st, "state": "stale"} if st["name"] == "simulation_plant" else st
            for st in ed.state.studies
        ]
        ed.show_results("simulation_plant")
        assert ed.state.results_note.startswith("The flowsheet has changed since this ran")

        ed.set_ask_replace(False)
        assert not ed.state.ask_replace and "could not save the preference" in ed.state.message
        ed.state.message = ""
        ed.set_ask_replace(False)  # no change: nothing to keep, nothing said
        assert ed.state.message == ""

        # A kind other than a simulation lists its result files.
        fisher = project / "studies" / "design"
        (fisher / "results").mkdir(parents=True)
        (fisher / "study.yaml").write_text(
            "study: {kind: fisher, free: [asm1.muH], observables: [{stream: tank}], "
            "t_obs: [1.0], sigma: 1.0}\n"
        )
        (fisher / "results" / "fisher.csv").write_text("")
        ed.state.studies = [
            *ed.state.studies,
            {"name": "design", "path": str(fisher), "state": "fresh"},
        ]
        ed.show_results("design")
        assert ed.state.results_mode == "files" and ed.state.results_files == ["fisher.csv"]

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
