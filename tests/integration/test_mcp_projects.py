"""``aquakin mcp``'s tools that change a project, through a real MCP client:
writing a flowsheet, and creating, editing, running and reading studies. Each
answer is compared with the aquakin function the editor's worker calls."""

from __future__ import annotations

import sys
import time

import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

import aquakin.study as st
from aquakin.mcp import runs
from aquakin.plant.spec import load_spec_file, write_spec
from tests.conftest import study_project
from tests.integration.mcp_client import error, value

#: Seconds a real half-day run of the one-tank plant may take (it compiles).
RUN_TIMEOUT = 300


@pytest.fixture
def project(tmp_path):
    return study_project(tmp_path)


def _finished(path) -> dict:
    """Wait for the run this process started last to end; the study's status after."""
    proc, _ = runs._STARTED[-1]
    assert proc.wait(RUN_TIMEOUT) is not None
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
    assert started["running"] is not None or started["state"] == "fresh"  # under way, or done
    status = _finished(path)
    assert status["state"] == "fresh" and status["has_results"] and status["running"] is None
    assert value("study_results", path=path) == st.study_results(path)

    # Results are not replaced unless asked; following the advice replaces them.
    assert error("run_study", path=path).endswith(
        "study new already has results; run it with replace=true to replace them, "
        "or copy_study first to keep them"
    )
    value("run_study", path=path, replace=True)
    assert _finished(path)["state"] == "fresh"


def test_a_study_not_running_has_no_run(project):
    status = value("study_status", path=str(project / "studies" / "short"))
    assert status == {
        "state": "not run",
        "has_results": False,
        "reasons": [],
        "notes": [],
        "running": None,
    }


def test_a_run_started_elsewhere_is_followed_and_cancelled(project, hold_run):
    path = project / "studies" / "short"
    held = hold_run(path)
    status = value("study_status", path=str(path))
    assert status["running"]["progress"]["message"] == "step 3"
    assert error("run_study", path=str(path)).endswith(
        "study short is running; run it once the run has ended, or cancel the run first"
    )
    assert value("cancel_study", path=str(path))["running"] is not None  # stops at its next report
    held.stdin.write("next\n")
    held.stdin.flush()
    assert held.stdout.readline() == "cancelled\n"
    assert held.wait(60) == 0
    assert error("cancel_study", path=str(path)).endswith("study short is not running")


def test_a_run_that_fails_is_recorded_with_aquakins_message(project):
    path = project / "studies" / "short"
    (path / "study.yaml").write_text("study:\n  kind: simulate\n  streams: [nowhere]\n")
    value("run_study", path=str(path))
    status = _finished(path)
    assert status["state"] == "failed"
    assert status["reasons"] == [
        "UnknownUnitError: \"Unknown unit 'nowhere' in endpoint 'nowhere'.\""
    ]


def test_a_run_refused_as_it_starts_says_why(project, hold_run, monkeypatch):
    """Another program may take the study between the server's look and the
    run's start: the run then refuses, and the assistant is told why."""
    import aquakin.study.running as running

    path = project / "studies" / "short"
    hold_run(path)
    monkeypatch.setattr(running, "refuse_while_running", lambda folder, action: None)
    message = error("run_study", path=str(path))
    assert message.endswith(
        "the run of study short did not start: error: study short is already running"
        f" (since {st.check_study(path).running['started']}); wait for it to end, or "
        "cancel it"
    )


def test_a_run_goes_on_after_the_server_exits(project):
    """The assistant's app closes the ``aquakin mcp`` process; the run it
    started finishes and records its results."""
    import asyncio

    from mcp import Client, StdioServerParameters

    path = project / "studies" / "short"
    server = StdioServerParameters(command=sys.executable, args=["-m", "aquakin", "mcp"])

    async def start():
        async with Client(server) as client:
            result = await client.call_tool("run_study", {"path": str(path)})
            assert not result.is_error, result.content[0].text

    asyncio.run(start())  # the server has exited when this returns
    end = time.monotonic() + RUN_TIMEOUT
    while st.check_study(path).state != "fresh":
        assert time.monotonic() < end, "the run did not finish after its server exited"
        time.sleep(0.1)
    assert st.check_study(path).running is None


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
