"""A worker for ``Editor`` tests that answers through aquakin's own ``serve`` in
this process, records every request, and can be told to refuse a command.

The editor takes its worker as an argument, so a test can count the requests
an edit causes (a cached check sends none) and make one command fail (the
refusal paths) without a subprocess. Answers are aquakin's: each request is
one JSON line through :func:`aquakin.__main__.serve`, as the real worker
reads it, and a refusal becomes a :class:`~aquakin_ui.worker.WorkerError`
with the message and type apart, as :class:`~aquakin_ui.worker.AquakinWorker`
makes it.
"""

from __future__ import annotations

import io
import json
import threading

from aquakin.__main__ import serve
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
