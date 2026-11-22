"""The worker client and a study run against stand-in processes, so each way a
process can answer (a stale reply, one reply then exit, events then an end)
can be forced."""

import gc
import sys
import textwrap

import pytest

from aquakin_ui.jobs import StudyRun
from aquakin_ui.worker import AquakinWorker, WorkerError


def _script(body: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(body)]


def test_a_reply_to_another_request_stops_the_worker():
    w = AquakinWorker(
        command=_script(
            """
            import sys
            for line in sys.stdin:
                print('{"id": 999, "ok": true, "result": 1}', flush=True)
            """
        )
    )
    with pytest.raises(WorkerError, match=r"^aquakin answered request 999, not 1$") as info:
        w.request("ping")
    assert info.value.type is None
    assert w.log_path is None  # stopped, and its log removed


def test_a_worker_that_exits_is_started_again_for_the_next_request():
    """Each stand-in answers one request (echoing its id) and exits; every
    request is still answered, by a fresh process."""
    w = AquakinWorker(
        command=_script(
            """
            import json, os, sys
            req = json.loads(sys.stdin.readline())
            print(json.dumps({"id": req["id"], "ok": True, "result": os.getpid()}), flush=True)
            """
        )
    )
    try:
        first = w.request("ping")
        w._proc.proc.wait(30)  # let it exit before the next request
        second = w.request("ping")
        assert first != second
    finally:
        w.stop()


def test_a_worker_that_dies_silently_says_only_that():
    w = AquakinWorker(command=_script("pass"))
    with pytest.raises(WorkerError, match=r"^aquakin exited while answering 'catalog'$"):
        w.request("catalog")


def test_a_refusal_without_a_message_is_still_an_error():
    w = AquakinWorker(
        command=_script(
            """
            import json, sys
            req = json.loads(sys.stdin.readline())
            print(json.dumps({"id": req["id"], "ok": False}), flush=True)
            sys.stdin.read()
            """
        )
    )
    try:
        with pytest.raises(WorkerError, match=r"^unknown error$"):
            w.request("ping")
    finally:
        w.stop()


_EVENTS = """
    import json
    print(json.dumps({"event": "progress", "fraction": 0.5, "message": "half"}), flush=True)
    print("a line that is not an event", flush=True)
    print(json.dumps({"event": "done", "state": "fresh", "has_results": True,
                      "reasons": [], "notes": []}), flush=True)
"""


def test_a_runs_events_are_passed_on_and_the_last_kept(tmp_path):
    events = []
    run = StudyRun(tmp_path, events.append, command=_script(_EVENTS))
    final = run.wait(30)
    done = {"event": "done", "state": "fresh", "has_results": True, "reasons": [], "notes": []}
    assert final == done
    assert events == [{"event": "progress", "fraction": 0.5, "message": "half"}, done]
    assert not run.running and not run.cancelled
    assert not run.log_path.exists()


def test_a_run_is_started_with_its_folder_and_replace_flag(tmp_path):
    run = StudyRun(tmp_path, lambda e: None, replace=True, command=_script(_EVENTS))
    assert run.command[-2:] == ["--replace", str(tmp_path)]
    run.wait(30)
    default = StudyRun(tmp_path, lambda e: None)
    try:
        assert default.command == [
            sys.executable,
            "-m",
            "aquakin",
            "study",
            "run",
            "--json",
            str(tmp_path),
        ]
    finally:
        default.cancel()
        default.wait(60)


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_cancelling_a_run_that_has_ended_changes_nothing(tmp_path):
    """Its input is closed: the cancel is dropped, then and when the pipe is
    collected (a "cancel" left in its buffer was raised again on collection)."""
    events = []
    run = StudyRun(tmp_path, events.append, command=_script(_EVENTS))
    final = run.wait(30)
    run.cancel()
    assert run.wait(1) == final and events[-1] == final
    del run
    gc.collect()


def test_a_run_is_running_until_its_process_ends(tmp_path):
    run = StudyRun(
        tmp_path,
        lambda e: None,
        command=_script(
            """
            import sys
            sys.stdin.readline()
            print('{"event": "cancelled"}', flush=True)
            """
        ),
    )
    assert run.running
    run.cancel()
    assert run.wait(30) == {"event": "cancelled"} and not run.running


def test_a_run_that_dies_after_progress_is_an_error_not_finished(tmp_path):
    """Progress is not an ending: a run whose last event is progress ended
    without a result."""
    events = []
    script = _script(
        """
        import json, sys
        print(json.dumps({"event": "progress", "fraction": 0.1, "message": "solving"}), flush=True)
        sys.stderr.write("Killed")
        """
    )
    run = StudyRun(tmp_path, events.append, command=script)
    final = run.wait(30)
    assert final == {"event": "error", "error": "the run ended without a result: Killed"}
    assert [e["event"] for e in events] == ["progress", "error"]
