"""The interface's line to aquakin: one long-lived ``aquakin serve`` process.

The interface never imports aquakin or JAX. It asks a worker process instead,
one JSON request per line, so the page starts without JAX and the solver never
loads the interface. Requests are serialized with a lock; replies are read on a
daemon thread so a timeout works on every platform. A timeout or a dead worker
stops the process, and the next request starts a fresh one. The worker's
standard error goes to a log file, whose last lines are quoted when it fails.
"""

from __future__ import annotations

import itertools
import json
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

#: Seconds to wait for one reply from a running worker.
TIMEOUT = 120.0
#: Extra seconds allowed for the first reply of a fresh worker (importing JAX).
START_TIMEOUT = 60.0


class WorkerError(RuntimeError):
    """The worker refused a request, timed out, or died."""


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
        self._proc = None
        self._replies: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._fresh = False
        self.log_path = Path(tempfile.gettempdir()) / "aquakin-ui-worker.log"

    def _start(self):
        flags = {}
        if sys.platform == "win32":
            flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            flags["start_new_session"] = True  # out of reach of the terminal's Ctrl-C
        log = open(self.log_path, "w")
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
            **flags,
        )
        log.close()
        self._fresh = True
        self._replies = queue.Queue()
        proc, replies = self._proc, self._replies

        def read():
            for line in proc.stdout:
                replies.put(line)
            replies.put(None)

        threading.Thread(target=read, daemon=True).start()

    def _log_tail(self, n: int = 5) -> str:
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
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def request(self, command: str, **fields):
        """Send one request and return its result.

        Parameters
        ----------
        command : str
            The worker command (``catalog``, ``models``, ``model``,
            ``check_unit``, ``check_spec``, ``ping``).
        **fields
            The command's fields.

        Returns
        -------
        object
            The command's result.

        Raises
        ------
        WorkerError
            If the worker answers with an error, times out, or exits.
        """
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            rid = next(self._ids)
            line = json.dumps({"id": rid, "command": command, **fields}) + "\n"
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except BrokenPipeError:
                self._start()
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            wait = self.timeout + (self.start_timeout if self._fresh else 0.0)
            try:
                raw = self._replies.get(timeout=wait)
            except queue.Empty:
                self._proc.kill()
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
                raise WorkerError(reply.get("error", "unknown error"))
            return reply["result"]
