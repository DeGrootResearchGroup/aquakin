"""A worker for ``Editor`` tests that answers through aquakin's own ``serve`` in
this process, records every request, and can be told to refuse a command; and a
study run that runs in this process too.

The editor takes its worker as an argument, so a test can count the requests
an edit causes (a cached check sends none) and make one command fail (the
refusal paths) without a subprocess. Answers are aquakin's: each request is
one JSON line through :func:`aquakin.__main__.serve`, as the real worker
reads it, and a refusal becomes a :class:`~aquakin_ui.worker.WorkerError`
with the message and type apart, as :class:`~aquakin_ui.worker.AquakinWorker`
makes it.

:class:`InProcessRun` stands in for :class:`~aquakin_ui.jobs.StudyRun` where a
test's subject is what the editor does with a run, not the run's process: it
runs aquakin's :func:`~aquakin.study.run_study` on a thread of the test process,
so the one-tank solve is compiled once per test process rather than once per
run (each real run imports JAX and compiles it afresh). Its events have the
shape ``aquakin study run --json`` writes; the tests that keep a real run
(``test_editor.py``: done and cancelled; ``test_editor_runs.py``: an error)
pin that shape against the process itself.
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

from aquakin.__main__ import serve
from aquakin.plant.errors import error_reply
from aquakin.progress import Cancelled
from aquakin.study import run_study
from aquakin_ui.worker import WorkerError


class InProcessWorker:
    """aquakin's worker, answered in this process.

    Attributes
    ----------
    sent : list of (str, dict)
        Every request, in order: its command and fields.
    refuse : dict
        ``{command: message}``: these commands fail with ``message`` (type
        ``"Refused"``) instead of being answered.
    """

    def __init__(self):
        self.sent: list[tuple[str, dict]] = []
        self.refuse: dict[str, str] = {}
        self._lock = threading.Lock()  # checks ask from the executor's threads

    def request(self, command: str, **fields):
        with self._lock:
            self.sent.append((command, fields))
            if command in self.refuse:
                raise WorkerError(self.refuse[command], "Refused")
            out = io.StringIO()
            serve(io.StringIO(json.dumps({"id": 1, "command": command, **fields}) + "\n"), out)
        reply = json.loads(out.getvalue())
        if not reply["ok"]:
            raise WorkerError(reply["error"], reply.get("type"))
        return reply["result"]

    def count(self, command: str) -> int:
        """How many ``command`` requests were sent."""
        return sum(1 for c, _ in self.sent if c == command)

    def stop(self) -> None:
        """Nothing to stop: there is no process."""


class InProcessRun:
    """A study run on a thread of this process, with ``StudyRun``'s interface.

    Parameters
    ----------
    folder : str or pathlib.Path
        The study folder.
    on_event : callable
        Called (on the run's thread) with each event, as ``StudyRun`` calls it.
    replace : bool, optional
        Replace the study's results.
    gate : threading.Event, optional
        The run waits for it to be set before it starts (``None``: no wait).
    """

    def __init__(
        self, folder, on_event, *, replace: bool = False, gate: threading.Event | None = None
    ):
        self.folder = Path(folder)
        self.on_event = on_event
        self.replace = replace
        self._gate = gate
        self.finished = threading.Event()
        self.final: dict | None = None
        self.cancelled = False  # a cancel was asked for
        threading.Thread(target=self._run, daemon=True).start()

    def _progress(self, p) -> None:
        if self.cancelled:
            raise Cancelled()
        self.on_event(
            {
                "event": "progress",
                "stage": p.stage,
                "done": p.done,
                "total": p.total,
                "fraction": p.fraction,
                "message": p.message,
            }
        )

    def _run(self) -> None:
        if self._gate is not None:
            self._gate.wait()
        try:
            status = run_study(self.folder, replace=self.replace, progress=self._progress)
        except Cancelled:
            final = {"event": "cancelled"}
        except Exception as exc:  # any failure is an error event, as the run process reports it
            final = {"event": "error", **error_reply(exc)}
        else:
            final = {
                "event": "done",
                "state": status.state,
                "has_results": status.has_results,
                "reasons": status.reasons,
                "notes": status.notes,
            }
        self.final = final
        self.on_event(final)
        self.finished.set()

    def cancel(self) -> None:
        """Stop the run at its next checkpoint."""
        self.cancelled = True

    def wait(self, timeout: float | None = None) -> dict | None:
        """Wait for the run to end; return its final event (``None`` on timeout)."""
        self.finished.wait(timeout)
        return self.final

    @property
    def running(self) -> bool:
        return not self.finished.is_set()
