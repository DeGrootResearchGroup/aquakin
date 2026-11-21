"""Running a study in a process of its own, with progress and cancel.

A study can run for minutes or hours, far beyond what the interactive worker
(:mod:`aquakin_ui.worker`) may spend on one request, and it must not hold up the
checks, Open or Save. So each run is its own ``python -m aquakin study run
--json <folder>`` process: it reports progress as JSON lines on standard
output and stops at its next checkpoint when it reads ``cancel`` on standard
input. Several studies can run at once.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

__all__ = ["StudyRun"]

#: Seconds a cancelled run may take to reach a checkpoint before it is killed.
CANCEL_GRACE = 30.0


class StudyRun:
    """One study run in its own process.

    Parameters
    ----------
    folder : str or pathlib.Path
        The study folder.
    on_event : callable
        Called (on a reader thread) with each event: ``{"event": "progress",
        "stage", "done", "total", "fraction", "message"}``, then exactly one of
        ``{"event": "done", "state", "reasons", "notes"}``, ``{"event":
        "cancelled"}`` or ``{"event": "error", "error"}``.
    replace : bool, optional
        Replace the study's results (the run refuses otherwise when it has some).
    command : list of str, optional
        The command to run (default ``python -m aquakin study run --json``).
    """

    def __init__(
        self,
        folder,
        on_event: Callable[[dict], None],
        *,
        replace: bool = False,
        command: list[str] | None = None,
    ):
        self.folder = Path(folder)
        self.on_event = on_event
        base = command or [sys.executable, "-m", "aquakin", "study", "run", "--json"]
        self.command = [*base, *(["--replace"] if replace else []), str(self.folder)]
        self.finished = threading.Event()
        self.final: dict | None = None
        self.cancelled = False  # a cancel was asked for
        fd, name = tempfile.mkstemp(prefix="aquakin-study-", suffix=".log")
        self.log_path = Path(name)
        log = os.fdopen(fd, "w")
        flags = {}
        if sys.platform == "win32":
            flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            flags["start_new_session"] = True
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
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self._proc.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "progress":
                self.final = event
            self.on_event(event)
        self._proc.wait()
        if self.final is None:
            # Asked to stop and killed before a checkpoint: that is a cancel (the
            # study keeps whatever results it had), not a failure.
            self.final = (
                {"event": "cancelled"}
                if self.cancelled
                else {
                    "event": "error",
                    "error": f"the run ended without a result{self._log_tail()}",
                }
            )
            self.on_event(self.final)
        self.finished.set()
        with contextlib.suppress(OSError):
            self.log_path.unlink()

    def _log_tail(self, n: int = 5) -> str:
        try:
            lines = self.log_path.read_text().strip().splitlines()[-n:]
        except OSError:
            return ""
        return (": " + " / ".join(lines)) if lines else ""

    def cancel(self) -> None:
        """Ask the run to stop at its next checkpoint; kill it if it has not
        within :data:`CANCEL_GRACE` seconds."""
        self.cancelled = True
        try:
            self._proc.stdin.write("cancel\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            # The run has ended. Close its input now: a "cancel" left in the
            # pipe's buffer would raise BrokenPipeError again when collected.
            with contextlib.suppress(OSError, ValueError):
                self._proc.stdin.close()

        def kill_later():
            if not self.finished.wait(CANCEL_GRACE):
                self._proc.kill()

        threading.Thread(target=kill_later, daemon=True).start()

    def wait(self, timeout: float | None = None) -> dict | None:
        """Wait for the run to end; return its final event (``None`` on timeout)."""
        self.finished.wait(timeout)
        return self.final

    @property
    def running(self) -> bool:
        return not self.finished.is_set()
