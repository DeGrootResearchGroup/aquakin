"""The worker client and a study run against stand-in processes, so each way a
process can answer (a stale reply, one reply then exit, events then an end)
can be forced, and each way one is discarded leaves nothing behind: the
process reaped and both of its pipes closed.

A pipe or process left open is reported by a ``ResourceWarning`` when it is
collected; here that is an error, and every test collects at its end so the
leak is charged to the test that made it."""

import gc
import io
import os
import signal
import subprocess
import sys
import textwrap
import threading
from typing import ClassVar

import pytest

from aquakin_ui import process, worker
from aquakin_ui.jobs import StudyRun
from aquakin_ui.worker import AquakinWorker, WorkerError

pytestmark = pytest.mark.filterwarnings(
    "error::ResourceWarning", "error::pytest.PytestUnraisableExceptionWarning"
)


@pytest.fixture(autouse=True)
def _collect_at_the_end():
    """Collect after each test, so a leaked pipe warns inside it."""
    yield
    gc.collect()


@pytest.fixture
def started(monkeypatch):
    """Every child the worker starts, kept so a test can look at it after the
    worker has let go of it."""
    kept = []
    real = worker.start_piped

    def record(*args, **kwargs):
        kept.append(real(*args, **kwargs))
        return kept[-1]

    monkeypatch.setattr(worker, "start_piped", record)
    return kept


def _discarded(piped) -> bool:
    """The process has been waited for and both pipes are closed."""
    return piped.proc.returncode is not None and piped.stdin.closed and piped.stdout.closed


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


def test_a_worker_that_exits_is_started_again_for_the_next_request(started):
    """Each stand-in answers one request (echoing its id) and exits; every
    request is still answered, by a fresh process, and the one that exited is
    discarded."""
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
        assert len(started) == 2 and _discarded(started[0])
    finally:
        w.stop()


def test_a_worker_that_stopped_reading_is_replaced_and_the_request_resent(tmp_path, started):
    """The first stand-in closes its input, then answers and stays alive, so the
    next request finds it running and its write breaks the pipe: a fresh
    process is given the same request, and the first is killed and reaped
    with both pipes closed rather than left running."""
    marker = tmp_path / "first"
    w = AquakinWorker(
        command=_script(
            f"""
            import json, os, sys, time
            marker = {str(marker)!r}
            fresh = not os.path.exists(marker)
            req = json.loads(sys.stdin.readline())
            if fresh:
                open(marker, "w").close()
                os.close(0)  # stop reading, and only then answer
            print(json.dumps({{"id": req["id"], "ok": True, "result": fresh}}), flush=True)
            if fresh:
                time.sleep(60)
            else:
                sys.stdin.read()
            """
        )
    )
    try:
        assert w.request("ping") is True
        first = started[0]
        assert first.proc.poll() is None  # alive, input closed
        assert w.request("ping") is False  # answered by the second process
        assert len(started) == 2 and w._proc is started[1]
        assert _discarded(first) and first.proc.returncode != 0  # killed
    finally:
        w.stop()


def test_stop_kills_a_worker_that_does_not_end_when_its_input_closes():
    """A stand-in that answers and then never reads again outlives the closed
    input; stop waits its few seconds, then kills it."""
    w = AquakinWorker(
        command=_script(
            """
            import json, sys, time
            req = json.loads(sys.stdin.readline())
            print(json.dumps({"id": req["id"], "ok": True, "result": 1}), flush=True)
            time.sleep(60)
            """
        )
    )
    assert w.request("ping") == 1
    piped = w._proc
    w.stop()
    assert w._proc is None and w.log_path is None
    assert _discarded(piped) and piped.proc.returncode != 0  # killed, not a clean exit


def test_stop_lets_a_worker_end_on_its_own_and_closes_both_pipes():
    """A worker that ends when its input closes exits cleanly, is reaped, and
    neither pipe is left open."""
    w = AquakinWorker(
        command=_script(
            """
            import json, sys
            for line in sys.stdin:
                req = json.loads(line)
                print(json.dumps({"id": req["id"], "ok": True, "result": 1}), flush=True)
            """
        )
    )
    assert w.request("ping") == 1
    piped = w._proc
    w.stop()
    assert _discarded(piped) and piped.proc.returncode == 0


def test_a_worker_that_times_out_is_reaped_with_both_pipes_closed(started):
    w = AquakinWorker(
        command=_script("import time; time.sleep(60)"), timeout=0.5, start_timeout=0.0
    )
    with pytest.raises(WorkerError, match=r"^aquakin did not answer 'ping' within 0.5 s$"):
        w.request("ping")
    assert w._proc is None and len(started) == 1
    assert _discarded(started[0]) and started[0].proc.returncode != 0
    w.stop()


def test_a_worker_that_dies_silently_says_only_that(started):
    """...and what is left of it is discarded: reaped, both pipes closed."""
    w = AquakinWorker(command=_script("pass"))
    with pytest.raises(WorkerError, match=r"^aquakin exited while answering 'catalog'$"):
        w.request("catalog")
    assert w._proc is None and len(started) == 1
    assert _discarded(started[0]) and started[0].proc.returncode == 0
    w.stop()


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
    assert _discarded(run._proc) and run._proc.proc.returncode == 0


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


def test_a_run_is_cleaned_up_before_it_is_reported_finished(tmp_path):
    """By the time ``finished`` is set (so ``wait`` returns and ``running`` is
    false) the process is reaped, both pipes are closed and the log removed:
    ``finished`` is swapped for an event that records the state when it is set,
    while the stand-in still waits for its go-ahead, so no ordering is left
    to chance."""
    run = StudyRun(
        tmp_path,
        lambda e: None,
        command=_script(
            """
            import sys
            sys.stdin.readline()
            print('{"event": "done", "state": "fresh"}', flush=True)
            """
        ),
    )
    seen = []

    class Recording(threading.Event):
        def set(self):
            seen.append((_discarded(run._proc), run.log_path.exists()))
            super().set()

    run.finished = Recording()
    run._proc.stdin.write("go\n")
    run._proc.stdin.flush()
    assert run.wait(30) == {"event": "done", "state": "fresh"}
    assert seen == [(True, False)]


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
    assert _discarded(run._proc)


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
    assert _discarded(run._proc)


class _FakePopen:
    """Records how it was called and writes a line to its standard error; a
    process that has already exited, with empty pipes."""

    calls: ClassVar[list] = []

    def __init__(self, command, **kwargs):
        _FakePopen.calls.append((command, kwargs))
        kwargs["stderr"].write("from the child\n")
        self.stdin, self.stdout = io.StringIO(), io.StringIO()
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        raise AssertionError("an exited process is not killed")


@pytest.mark.parametrize(
    ("platform", "creationflags", "new_session"),
    [("linux", 0, True), ("darwin", 0, True), ("win32", 0x200, False)],
)
def test_a_child_gets_a_session_or_on_windows_a_process_group_of_its_own(
    monkeypatch, platform, creationflags, new_session
):
    """The exact arguments each platform passes to ``Popen``: a new session on
    POSIX, a new process group (and no session, which Windows lacks) on
    Windows; ``CREATE_NEW_PROCESS_GROUP`` is supplied where the platform has
    none."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(_FakePopen, "calls", [])
    seen = []
    piped = process.start_piped(["prog", "arg"], "aquakin-test-", seen.append)
    try:
        piped.close()
        assert seen == [piped] and piped.stdin.closed and piped.stdout.closed
        [(command, kwargs)] = _FakePopen.calls
        log = kwargs.pop("stderr")
        assert command == ["prog", "arg"]
        assert kwargs == {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
            "start_new_session": new_session,
            "creationflags": creationflags,
        }
        assert log.closed and piped.log_path.read_text() == "from the child\n"
        assert piped.log_path.name.startswith("aquakin-test-")
    finally:
        piped.log_path.unlink()


@pytest.mark.skipif(sys.platform == "win32", reason="kills the left-over process with SIGKILL")
def test_close_leaves_stdout_to_a_reader_still_waiting_for_its_end(monkeypatch):
    """A process the child started keeps its standard output open after the
    child is killed, so the reader cannot reach the end: ``close`` waits
    ``READER_TIMEOUT`` for it and returns with ``stdout`` open (closing a pipe
    another thread is reading could block); the reader closes it once the
    pipe's last holder is gone."""
    monkeypatch.setattr(process, "READER_TIMEOUT", 0.2)
    pids, told = [], threading.Event()

    def read(piped):
        for line in piped.stdout:
            pids.append(int(line))
            told.set()

    piped = process.start_piped(
        _script(
            """
            import subprocess, sys
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            print(child.pid, flush=True)
            sys.stdin.read()
            """
        ),
        "aquakin-test-",
        read,
    )
    try:
        assert told.wait(30)
        piped.close()
        assert piped.proc.returncode is not None and piped.stdin.closed
        assert piped.reader.is_alive() and not piped.stdout.closed
    finally:
        if pids:
            os.kill(pids[0], signal.SIGKILL)
        piped.reader.join(30)
        piped.log_path.unlink()
    assert not piped.reader.is_alive() and piped.stdout.closed


def test_close_from_the_reader_itself_and_again_is_harmless():
    """A reader that discards its own process once it has read everything
    (as a study run does) closes both pipes without joining itself, and a
    second ``close`` changes nothing."""
    lines = []

    def read(piped):
        lines.extend(piped.stdout)
        piped.close()
        lines.append(piped.stdout.closed)

    piped = process.start_piped(_script("print('one')"), "aquakin-test-", read)
    piped.reader.join(30)
    piped.close()
    piped.log_path.unlink()
    assert lines == ["one\n", True] and _discarded(piped)
