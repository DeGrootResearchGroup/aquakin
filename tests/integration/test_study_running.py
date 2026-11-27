"""A study's run in progress, kept in its folder (``aquakin/study/running.py``):
one run at a time from any program, its progress readable by any program, a
cancel from any program, and no lock left behind by a run that was killed."""

from __future__ import annotations

import re
import signal

import pytest

import aquakin.study as st
from aquakin.__main__ import main
from aquakin.progress import Cancelled
from aquakin.study import running
from aquakin.study.common import CANCEL_FILE, RUNNING_FILE
from tests.conftest import study_project

DYNAMIC = {
    "kind": "simulate",
    "time": {"t_span": [0, 0.5], "t_eval": {"start": 0, "stop": 0.5, "num": 3}},
    "streams": ["tank"],
}


@pytest.fixture
def study(tmp_path):
    """The one-tank project's ``short`` study folder."""
    return study_project(tmp_path) / "studies" / "short"


@pytest.fixture
def held(study, hold_run):
    """Another process holding ``study``'s run lock, at its first progress report."""
    return hold_run(study)


def test_a_run_elsewhere_is_seen_and_refuses_a_second_run(study, held):
    state = st.check_study(study)
    assert state.state == "not run"  # the standing of the results before the run
    assert state.running["progress"] == {
        "stage": "fit",
        "done": 3,
        "total": 10,
        "fraction": 0.3,
        "message": "step 3",
    }
    assert state.running["pid"] == held.pid
    started = state.running["started"]
    assert st.list_studies(study.parents[1])[0]["running"] == state.running
    with pytest.raises(st.StudyError, match=re.escape(f"study short is already running (since {started})")):
        st.run_study(study)
    for action, call in [
        ("freeze", lambda: st.freeze_study(study)),
        ("unfreeze", lambda: st.unfreeze_study(study)),
        ("edit", lambda: st.update_study(study, DYNAMIC)),
    ]:
        with pytest.raises(st.StudyError, match=f"study short is running; {action} it once"):
            call()


def test_a_copy_of_a_running_study_is_not_running(study, held):
    copy = st.copy_study(study, "copy")
    assert not (copy / RUNNING_FILE).exists()
    assert st.check_study(copy).running is None


def test_a_run_that_ends_releases_the_study(study, held):
    held.stdin.close()
    assert held.wait(60) == 0
    assert not (study / RUNNING_FILE).exists()
    assert st.check_study(study).running is None
    assert st.run_study(study).state == "fresh"


def test_a_killed_run_leaves_no_lock(study, held):
    held.send_signal(signal.SIGKILL)
    held.wait(60)
    assert (study / RUNNING_FILE).exists()  # left behind, but nobody holds it
    assert st.check_study(study).running is None
    status = st.run_study(study)
    assert status.state == "fresh" and status.running is None


def test_cancel_reaches_a_run_started_elsewhere(study, held):
    held.stdin.write("next\n")
    held.stdin.flush()
    assert held.stdout.readline() == "reported\n"
    assert st.check_study(study).running["progress"]["message"] == "step 4"
    st.cancel_study(study)
    assert (study / CANCEL_FILE).exists()
    held.stdin.write("next\n")
    held.stdin.flush()
    assert held.stdout.readline() == "cancelled\n"  # stopped at its next report
    assert held.wait(60) == 0
    assert not (study / CANCEL_FILE).exists() and not (study / RUNNING_FILE).exists()


def test_a_cancelled_run_stops_at_its_next_checkpoint(study):
    st.update_study(study, DYNAMIC)
    seen = []

    def progress(p):
        # The file holds this report before the caller's callback sees it.
        seen.append((p.stage, st.check_study(study).running["progress"]["stage"]))
        if len(seen) == 1:
            st.cancel_study(study)

    with pytest.raises(Cancelled, match="^$"):  # Cancelled carries no message
        st.run_study(study, progress=progress)
    assert seen[0][0] == seen[0][1] and len(seen) == 1
    status = st.check_study(study)
    assert status.state == "cancelled" and status.running is None
    assert not (study / CANCEL_FILE).exists() and not (study / RUNNING_FILE).exists()
    # An old cancel request does not stop the next run.
    (study / CANCEL_FILE).touch()
    assert st.run_study(study).state == "fresh"


def test_cancelling_a_study_that_is_not_running_is_refused(study):
    with pytest.raises(st.StudyError, match="study short is not running"):
        st.cancel_study(study)


def test_an_unreadable_running_file_still_counts_as_running(study, held):
    (study / RUNNING_FILE).write_text("{{ half written")
    assert st.check_study(study).running == {"started": None, "pid": None, "progress": None}
    with pytest.raises(st.StudyError, match=r"study short is already running; wait"):
        st.run_study(study)


def test_a_lock_on_a_file_since_removed_is_taken_again(study, monkeypatch):
    """The run before may remove ``running.yaml`` between this run's open and
    its lock: the lock is then on a file nobody sees, and is taken again."""
    calls = []
    real = running.os.path.samestat

    def samestat(a, b):
        calls.append(1)
        return False if len(calls) == 1 else real(a, b)

    monkeypatch.setattr(running.os.path, "samestat", samestat)
    with running.RunLock(study):
        assert len(calls) == 2
        assert running.run_state(study) is not None


def test_the_command_line_checks_and_cancels_a_run(study, held, capsys):
    assert main(["study", "check", str(study)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "not run"
    assert out.splitlines()[1].startswith("  running since ") and out.endswith(": step 3\n")
    assert main(["study", "cancel", str(study)]) == 0
    assert capsys.readouterr().out == "asked to stop at its next checkpoint\n"
    assert (study / CANCEL_FILE).exists()
    held.stdin.close()
    held.wait(60)
    assert main(["study", "cancel", str(study)]) == 2
    assert "study short is not running" in capsys.readouterr().err
