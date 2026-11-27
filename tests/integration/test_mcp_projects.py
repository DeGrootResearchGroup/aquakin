"""``aquakin mcp``'s tools that change a project, through a real MCP client:
writing a flowsheet, and creating, editing, running and reading studies. Each
answer is compared with the aquakin function the editor's worker calls."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

import aquakin.study as st
from aquakin.mcp import projects
from aquakin.plant.spec import load_spec_file, write_spec
from tests.conftest import study_project
from tests.integration.mcp_client import error, value

#: Seconds a real half-day run of the one-tank plant may take (it compiles).
RUN_TIMEOUT = 300


@pytest.fixture
def project(tmp_path):
    return study_project(tmp_path)


def _finished(path) -> dict:
    """Wait for the run of ``path`` this server started; its status after."""
    run = projects.RUNS.get(path)
    assert run is not None and run.wait(RUN_TIMEOUT), "the run did not end"
    return value("study_status", path=str(path))


def test_write_flowsheet_writes_what_the_editor_writes_and_refuses_to_overwrite(project):
    spec = load_spec_file(project / "flowsheet.yaml")
    target = project / "second.yaml"
    written = value("write_flowsheet", path=str(target), spec=spec)
    assert written == load_spec_file(target)
    assert written == write_spec(spec, project / "third.yaml")
    message = error("write_flowsheet", path=str(target), spec=spec)
    assert message.endswith(
        f"FileExistsError: {target} already exists; read it with read_flowsheet, then "
        f"write it again with replace=true to overwrite it"
    )
    # Following the advice: read it, write it again with replace.
    spec["units"]["tank"]["volume"] = 1500.0
    value("write_flowsheet", path=str(target), spec=spec, replace=True)
    assert load_spec_file(target)["units"]["tank"]["volume"] == 1500.0


def test_a_flowsheet_that_does_not_check_out_writes_nothing(project):
    spec = load_spec_file(project / "flowsheet.yaml")
    spec["units"]["tank"]["volume"] = "lots"
    target = project / "broken.yaml"
    assert error("write_flowsheet", path=str(target), spec=spec).endswith(
        "SpecError: units.tank: 'volume' must be a real number, got 'lots'"
    )
    assert not target.exists()


def test_reading_tools_answer_as_aquakin_study_does(project):
    assert value("list_studies", project=str(project)) == st.list_studies(project)
    assert value("study_settings") == st.study_form()
    flowsheet = project / "flowsheet.yaml"
    assert value("study_choices", flowsheet=str(flowsheet)) == st.plant_choices(flowsheet)
    short = project / "studies" / "short"
    assert value("read_study", path=str(short)) == st.read_study(short)


def test_a_study_is_created_updated_run_and_read(project):
    study = {"kind": "simulate", "streams": ["tank"], "description": "a new one"}
    path = value("create_study", project=str(project), name="new", study=study)
    assert path == str(project / "studies" / "new")
    assert value("read_study", path=path)["description"] == "a new one"

    updated = st.read_study(path) | {"description": "edited"}
    assert value("update_study", path=path, study=updated) == path
    assert st.read_study(path)["description"] == "edited"

    started = value("run_study", path=path)
    assert started["state"] == "not run"
    status = _finished(path)
    assert status["outcome"] == "finished" and not status["running"]
    assert status["state"] == "fresh" and status["has_results"]
    assert status["progress"]["stage"] == "simulate"  # the last report the run made
    assert value("study_results", path=path) == st.study_results(path)

    # Results are not replaced unless asked; following the advice replaces them.
    assert error("run_study", path=path).endswith(
        "study new already has results; run it with replace=true to replace them, "
        "or copy_study first to keep them"
    )
    value("run_study", path=path, replace=True)
    assert _finished(path)["outcome"] == "finished"


def test_a_study_not_run_here_has_no_run_fields(project):
    status = value("study_status", path=str(project / "studies" / "short"))
    assert status == {
        "state": "not run",
        "has_results": False,
        "reasons": [],
        "notes": [],
        "running": False,
    }


class _HeldRun:
    """A stand-in for :func:`aquakin.study.run_study` that reports progress and
    waits at a checkpoint until released, so a test can act while it runs."""

    def __init__(self):
        self.at_checkpoint = threading.Event()
        self.release = threading.Event()

    def __call__(self, path, *, replace=False, progress=None):
        from aquakin.progress import Progress

        progress(Progress("simulate", 0, 2, "starting"))
        self.at_checkpoint.set()
        assert self.release.wait(RUN_TIMEOUT)
        progress(Progress("simulate", 1, 2, "halfway"))  # a cancel stops it here
        return st.check_study(path)


def test_a_running_study_reports_progress_and_can_be_cancelled(project, monkeypatch):
    held = _HeldRun()
    monkeypatch.setattr(st, "run_study", held)
    path = str(project / "studies" / "short")
    value("run_study", path=path)
    assert held.at_checkpoint.wait(RUN_TIMEOUT)
    status = value("study_status", path=path)
    assert status["running"] is True
    assert status["progress"] == {
        "stage": "simulate",
        "done": 0,
        "total": 2,
        "fraction": 0.0,
        "message": "starting",
    }
    assert error("run_study", path=path).endswith(
        "study short is already running; ask study_status how it is going, or "
        "cancel_study to stop it"
    )
    assert value("cancel_study", path=path)["running"] is True  # stops at its next checkpoint
    held.release.set()
    assert _finished(path)["outcome"] == "cancelled"
    assert error("cancel_study", path=path).endswith("study short is not running here")


def test_a_run_that_fails_reports_aquakins_message(project, monkeypatch):
    def failing(path, *, replace=False, progress=None):
        raise st.StudyError("the plant did not converge")

    monkeypatch.setattr(st, "run_study", failing)
    path = str(project / "studies" / "short")
    value("run_study", path=path)
    status = _finished(path)
    assert status["outcome"] == "failed"
    assert status["error"] == "the plant did not converge"
    assert status["type"] == "StudyError"


def test_cancelling_a_study_never_started_here_is_refused(project):
    path = project / "studies" / "short"
    assert error("cancel_study", path=str(path)).endswith("study short is not running here")


def test_copy_freeze_and_unfreeze(project):
    short = project / "studies" / "short"
    copy = value("copy_study", path=str(short), name="kept")
    assert copy == str(project / "studies" / "kept")
    assert value("freeze_study", path=copy)["state"] == "not run"
    assert st.check_study(copy).frozen
    assert not value("unfreeze_study", path=copy)["running"]
    assert not st.check_study(copy).frozen
    names = [s["name"] for s in value("list_studies", project=str(project))]
    assert names == ["kept", "short"]


def test_unfreezing_a_study_with_results_needs_discard_results(project):
    short = project / "studies" / "short"
    st.run_study(short)
    st.freeze_study(short)
    assert "discard_results=True" in error("unfreeze_study", path=str(short))
    assert st.check_study(short).frozen
    status = value("unfreeze_study", path=str(short), discard_results=True)
    assert not status["has_results"] and not st.check_study(short).frozen
