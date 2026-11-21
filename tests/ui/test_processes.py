"""The worker client and a study run against stand-in processes, so each way a
process can answer (a stale reply, one reply then exit, events then an end)
can be forced."""

import gc
import sys
import textwrap

import pytest

from aquakin_ui.jobs import StudyRun


def _script(body: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(body)]


_EVENTS = """
    import json
    print(json.dumps({"event": "progress", "fraction": 0.5, "message": "half"}), flush=True)
    print("a line that is not an event", flush=True)
    print(json.dumps({"event": "done", "state": "fresh", "has_results": True,
                      "reasons": [], "notes": []}), flush=True)
"""


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
