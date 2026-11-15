"""The interface's client for the ``aquakin serve`` worker, against a real worker."""

import sys

import pytest

from aquakin_ui.worker import AquakinWorker, WorkerError


@pytest.fixture(scope="module")
def worker():
    w = AquakinWorker()
    yield w
    w.stop()


def test_requests_and_refusals(worker):
    assert "version" in worker.request("ping")
    with pytest.raises(WorkerError, match="unknown command"):
        worker.request("nope")
    assert worker.request("check_unit", kind="splitter", name="s", settings={"model": "asm1"})["ok"]


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
