"""A study run in its own process (``aquakin_ui.jobs.StudyRun``), with stand-in
commands for the run so the endings can be forced."""

import sys

from aquakin_ui import jobs
from aquakin_ui.jobs import StudyRun


def test_a_run_killed_after_a_cancel_is_reported_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "CANCEL_GRACE", 0.5)
    events = []
    # A run that never reaches a checkpoint (it never reads its input).
    run = StudyRun(
        tmp_path, events.append, command=[sys.executable, "-c", "import time; time.sleep(60)"]
    )
    run.cancel()
    assert run.wait(30) == {"event": "cancelled"} and events == [{"event": "cancelled"}]


def test_a_run_that_ends_without_a_result_is_an_error(tmp_path):
    run = StudyRun(
        tmp_path,
        lambda event: None,
        command=[sys.executable, "-c", "import sys; print('not json'); sys.stderr.write('boom')"],
    )
    final = run.wait(30)
    assert final["event"] == "error" and "ended without a result: boom" in final["error"]
