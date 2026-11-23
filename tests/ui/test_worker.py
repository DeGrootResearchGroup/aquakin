"""The interface's client for the ``aquakin serve`` worker, against a real worker."""

import sys

import pytest

from aquakin_ui.worker import AquakinWorker, WorkerError

#: A stand-in worker that answers every request with its own id, until its input closes.
_ECHO = [
    sys.executable,
    "-c",
    "import json, sys\n"
    "for line in sys.stdin:\n"
    "    req = json.loads(line)\n"
    "    print(json.dumps({'id': req['id'], 'ok': True, 'result': 1}), flush=True)\n",
]


def test_requests_and_refusals(real_worker):
    assert "version" in real_worker.request("ping")
    with pytest.raises(WorkerError, match="unknown command"):
        real_worker.request("nope")
    assert real_worker.request(
        "check_unit", kind="splitter", name="s", settings={"model": "asm1"}
    )["ok"]


def test_a_worker_that_does_not_answer_times_out():
    w = AquakinWorker(
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=0.5,
        start_timeout=0.5,
    )
    with pytest.raises(WorkerError, match="did not answer"):
        w.request("ping")


def test_a_worker_that_dies_is_reported_with_its_error_output():
    w = AquakinWorker(command=[sys.executable, "-c", "import sys; sys.stderr.write('boom\\n')"])
    with pytest.raises(WorkerError, match=r"exited(?s:.*)boom"):
        w.request("ping")


def test_each_worker_has_its_own_log_removed_when_it_stops():
    a, b = AquakinWorker(command=_ECHO), AquakinWorker(command=_ECHO)
    try:
        a.request("ping")
        b.request("ping")
        assert a.log_path != b.log_path and a.log_path.exists() and b.log_path.exists()
        path = a.log_path
    finally:
        a.stop()
        b.stop()
    assert not path.exists()
