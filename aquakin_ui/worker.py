"""The interface's line to aquakin: one long-lived ``aquakin serve`` process.

The interface never imports aquakin or JAX. It asks a worker process instead,
one JSON request per line, so the page starts without JAX and the solver never
loads the interface. Requests are serialized with a lock; replies are read on a
daemon thread so a timeout works on every platform. A timeout or a dead worker
stops the process, and the next request starts a fresh one. The worker's
standard error goes to a log file, whose last lines are quoted when it fails.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import queue
import sys
import threading
from pathlib import Path
from typing import IO

from aquakin_ui.process import Piped, start_piped

#: Seconds to wait for one reply from a running worker.
TIMEOUT = 120.0
#: Extra seconds allowed for the first reply of a fresh worker (importing JAX).
START_TIMEOUT = 60.0


class WorkerError(RuntimeError):
    """The worker refused a request, timed out, or died.

    Parameters
    ----------
    message : str
        What went wrong, as a user may read it (no exception class in front).
    type : str, optional
        The class name of aquakin's exception when the worker refused the
        request (``"StudyError"``, ``"SpecError"``, ...); ``None`` when there
        was none (a timeout, a dead worker, an unknown command).
    """

    def __init__(self, message: str, type: str | None = None):
        super().__init__(message)
        self.type = type


class AquakinWorker:
    """A client for ``python -m aquakin serve``.

    Parameters
    ----------
    command : list of str, optional
        The worker's command line (default this interpreter's ``-m aquakin serve``).
    timeout : float, optional
        Seconds to wait for one reply.
    start_timeout : float, optional
        Extra seconds for the first reply after the worker starts.
    """

    def __init__(
        self, command=None, timeout: float = TIMEOUT, start_timeout: float = START_TIMEOUT
    ):
        self.command = command or [sys.executable, "-m", "aquakin", "serve"]
        self.timeout = timeout
        self.start_timeout = start_timeout
        self._proc: Piped | None = None
        self._replies: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._fresh = False
        self.log_path: Path | None = None

    def _start(self) -> Piped:
        # One log per worker, so two editors never share (or truncate) a log.
        self._remove_log()
        proc = start_piped(self.command, "aquakin-ui-worker-")
        self._proc = proc
        self.log_path = proc.log_path
        self._fresh = True
        self._replies = queue.Queue()
        stdout, replies = proc.stdout, self._replies

        def read():
            for line in stdout:
                replies.put(line)
            replies.put(None)

        threading.Thread(target=read, daemon=True).start()
        return proc

    @staticmethod
    def _send(stdin: IO[str], line: str) -> None:
        stdin.write(line)
        stdin.flush()

    def _remove_log(self):
        if self.log_path is not None:
            with contextlib.suppress(OSError):
                self.log_path.unlink()

    def _log_tail(self, n: int = 5) -> str:
        if self.log_path is None:
            return ""
        try:
            lines = self.log_path.read_text().strip().splitlines()[-n:]
        except OSError:
            return ""
        return ("\n" + "\n".join(lines)) if lines else ""

    def stop(self):
        """Stop the worker (closing its input ends its loop)."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.proc.wait(timeout=5)
            except Exception:
                self._proc.proc.kill()
            self._proc = None
        self._remove_log()
        self.log_path = None

    def request(self, command: str, **fields):
        """Send one request and return its result.

        Parameters
        ----------
        command : str
            The worker command (one of ``aquakin serve``'s; see
            :mod:`aquakin.__main__`).
        **fields
            The command's fields.

        Returns
        -------
        object
            The command's result.

        Raises
        ------
        WorkerError
            If the worker answers with an error (its ``type`` set to the
            exception's class name), times out, or exits.
        """
        with self._lock:
            proc = self._proc
            if proc is None or proc.proc.poll() is not None:
                proc = self._start()
            rid = next(self._ids)
            line = json.dumps({"id": rid, "command": command, **fields}) + "\n"
            try:
                self._send(proc.stdin, line)
            except BrokenPipeError:
                proc = self._start()
                self._send(proc.stdin, line)
            wait = self.timeout + (self.start_timeout if self._fresh else 0.0)
            try:
                raw = self._replies.get(timeout=wait)
            except queue.Empty:
                proc.proc.kill()
                self._proc = None
                raise WorkerError(
                    f"aquakin did not answer '{command}' within {wait:g} s{self._log_tail()}"
                ) from None
            self._fresh = False
            if raw is None:
                self._proc = None
                raise WorkerError(f"aquakin exited while answering '{command}'{self._log_tail()}")
            reply = json.loads(raw)
            if reply.get("id") != rid:
                self.stop()
                raise WorkerError(f"aquakin answered request {reply.get('id')}, not {rid}")
            if not reply.get("ok"):
                raise WorkerError(reply.get("error", "unknown error"), reply.get("type"))
            return reply["result"]
