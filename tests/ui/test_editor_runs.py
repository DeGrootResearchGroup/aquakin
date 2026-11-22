"""The Studies tab's refusals, no-ops and run events, driven through an
``Editor`` without a browser."""

import asyncio
import itertools
import threading

import pytest
from trame.app import get_server

from aquakin_ui.app import Editor

from .conftest import study_project
from .inprocess import InProcessWorker

_names = itertools.count()


@pytest.fixture
def project(tmp_path):
    """The one-tank project, with a second study, ``broken``, whose stream does
    not exist: it builds, and its run fails."""
    project = study_project(tmp_path)
    broken = project / "studies" / "broken"
    broken.mkdir()
    (broken / "study.yaml").write_text(
        "study:\n  kind: simulate\n  time: {t_span: [0, 0.5]}\n  streams: [nope]\n"
    )
    return project


@pytest.fixture
def editor(project):
    server = get_server(name=f"aquakin-ui-runs-{next(_names)}", client_type="vue3")
    worker = InProcessWorker()
    ed = Editor(server, worker=worker, path=project / "flowsheet.yaml")
    yield ed, worker
    ed.stop_runs()
    for run in list(ed._runs.values()):
        run.wait(60)


def test_actions_on_a_study_that_is_not_listed_send_nothing(editor):
    ed, worker = editor
    sent = len(worker.sent)
    ed.run_study("nope")
    ed.freeze_study("nope")
    ed.unfreeze_study("nope")
    ed.copy_study("nope")
    ed.cancel_run("nope")
    assert worker.sent[sent:] == [] and ed._runs == {}
    assert not ed.state.study_confirm_open


def test_a_copy_takes_the_first_free_name(editor):
    ed, worker = editor
    ed.copy_study("short")
    ed.copy_study("short")
    names = [st["name"] for st in ed.state.studies]
    assert names == ["broken", "short", "short_copy", "short_copy2"]
    assert [f["name"] for c, f in worker.sent if c == "copy_study"] == ["short_copy", "short_copy2"]


def test_a_refused_study_command_is_reported_by_name(editor):
    ed, worker = editor
    worker.refuse["freeze_study"] = "the folder is read-only"
    ed.freeze_study("short")
    assert ed.state.message == "freeze study failed: the folder is read-only"
    assert not ed._study("short")["frozen"]


def test_a_study_list_the_worker_refuses_keeps_the_last_one(editor):
    ed, worker = editor
    listed = list(ed.state.studies)
    worker.refuse["list_studies"] = "permission denied"
    ed.refresh_studies()
    assert ed.state.message == "could not list the studies: permission denied"
    assert ed.state.studies == listed


def test_an_unreadable_form_edit_holds_a_run_back(editor):
    """Run carries the form's unapplied edits; when they cannot be read the run
    waits (and the field is named) rather than running the old settings."""
    ed, _ = editor
    ed.on_select({"nodes": ["tank"], "edges": []})
    rows = [{**r, "value": "big"} if r["key"] == "volume" else r for r in ed.state.sel_rows]
    ed.run_study("short", pending=rows)
    assert ed._runs == {} and set(ed.state.sel_errors) == {"volume"}


def test_a_question_answered_after_its_study_went_does_nothing(editor):
    """The study is listed with results, so Run asks; the list is refreshed
    without it (its folder deleted meanwhile) before the answer comes."""
    ed, worker = editor
    ed.state.studies = [
        {**st, "has_results": True} if st["name"] == "short" else st for st in ed.state.studies
    ]
    ed.run_study("short")
    assert ed.state.study_confirm_open
    ed.state.studies = [st for st in ed.state.studies if st["name"] != "short"]
    sent = len(worker.sent)
    ed.answer_study("go")
    assert ed._runs == {} and worker.sent[sent:] == []


def test_a_failed_run_is_reported_and_its_events_applied_on_the_pages_loop(editor):
    """A run's events arrive on its reader thread; with a page served they are
    applied on the page's loop. A run that fails reports aquakin's message."""
    ed, _ = editor
    applied = []
    original = ed._apply_run_event
    ed._apply_run_event = lambda name, event: (
        applied.append((event["event"], threading.current_thread())),
        original(name, event),
    )

    async def go():
        ed.run_study("broken")
        assert ed._study("broken")["running"]
        loop = asyncio.get_running_loop()
        end = loop.time() + 300
        while "broken" in ed._runs and loop.time() < end:
            await asyncio.sleep(0.05)

    asyncio.run(go())
    assert applied and applied[-1][0] == "error"
    assert {t for _, t in applied} == {threading.main_thread()}
    assert ed.state.message.startswith("broken failed: Unknown stream 'nope'")
    assert "KeyError" not in ed.state.message
    assert "broken" not in ed.state.study_progress


def test_closing_the_editor_cancels_every_run(editor):
    ed, _ = editor
    ed.run_study("short")
    run = ed._runs["short"]
    ed.stop_runs()
    assert run.cancelled
    assert run.wait(300)["event"] == "cancelled"
    assert ed.state.message == "short: cancelled"
