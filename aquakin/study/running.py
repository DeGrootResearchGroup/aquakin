"""A study's run in progress, kept in the study folder so every program sees it.

A run (:func:`aquakin.study.run_study`, from the command line, the editor or an
AI assistant through ``aquakin mcp``) holds ``running.yaml`` in the study folder
for as long as it lasts, under an operating-system lock, and writes into it
when it started and its latest progress. So:

- a second run of the same study, from any program, is refused while the first
  is running;
- any program can say that a study is running and how far it has got
  (:func:`run_state`), and ask it to stop (:func:`cancel_study`, which creates
  ``cancel``; the run stops at its next checkpoint);
- a run that crashes or is killed leaves no lock behind: the operating system
  releases it with the process, and a ``running.yaml`` nobody holds is ignored.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

from aquakin.study.common import CANCEL_FILE, RUNNING_FILE, StudyError, _now

#: Where the lock sits on Windows, whose locks bar other handles from reading
#: the bytes they cover: one byte far past what the file holds, so readers can
#: still read its contents.
_LOCK_OFFSET = 1 << 30
#: Attempts, and seconds between them, to take a lock another program holds
#: only for a moment (:func:`run_state` takes it briefly to see if it is free).
_ATTEMPTS, _PAUSE = 20, 0.05


def _try_lock(fd: int) -> bool:
    """Take the run lock on an open file without waiting; whether it was free."""
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def _read(path: Path) -> dict:
    """What a running study's file says (``{}`` while it is being rewritten)."""
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def run_state(folder) -> dict | None:
    """Whether a study is running, from whichever program, and how far it has got.

    Parameters
    ----------
    folder : str or pathlib.Path
        The study folder.

    Returns
    -------
    dict or None
        ``None`` when no run holds the study; otherwise ``{"started",
        "progress"}``: when the run started (ISO 8601, UTC) and its latest
        progress report (``{"stage", "done", "total", "fraction",
        "message"}``, ``None`` before the first).
    """
    path = Path(folder) / RUNNING_FILE
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return None
    try:
        if _try_lock(fd):
            _unlock(fd)
            return None  # left by a run that was killed: nobody holds it
    finally:
        os.close(fd)
    data = _read(path)
    return {"started": data.get("started"), "progress": data.get("progress")}


def refuse_while_running(folder, action: str) -> None:
    """Raise if a run holds the study: ``action`` would change what it is running.

    Raises
    ------
    StudyError
        Naming the study and the action, if it is running.
    """
    if run_state(folder) is not None:
        raise StudyError(
            f"study {Path(folder).name} is running; {action} it once the run has "
            f"ended, or cancel the run first"
        )


def cancel_study(path) -> None:
    """Ask a study's run to stop at its next checkpoint, whichever program started it.

    The study keeps the results it had before the run.

    Parameters
    ----------
    path : str or pathlib.Path
        The study folder.

    Raises
    ------
    StudyError
        If the study is not running.
    """
    folder = Path(path)
    if run_state(folder) is None:
        raise StudyError(f"study {folder.name} is not running")
    (folder / CANCEL_FILE).touch()


class RunLock:
    """The run of one study, held from start to end (a context manager).

    Entering takes the study's lock, refusing if another run holds it, clears
    an old ``cancel`` request and records when the run started. :meth:`report`
    records progress and stops the run when it has been cancelled. Leaving
    removes both files and releases the lock, however the run ended.

    Parameters
    ----------
    folder : pathlib.Path
        The study folder.
    """

    def __init__(self, folder: Path):
        self.folder = folder
        self.path = folder / RUNNING_FILE
        self._fd: int | None = None
        self._started = _now()

    def __enter__(self) -> RunLock:
        for _ in range(_ATTEMPTS):
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
            if not _try_lock(fd):
                os.close(fd)
                time.sleep(_PAUSE)
                continue
            # The run that held the lock may have removed the file between our
            # open and our lock: then this lock is on a file nobody else sees.
            if os.path.exists(self.path) and os.path.samestat(os.fstat(fd), os.stat(self.path)):
                self._fd = fd
                break
            _unlock(fd)
            os.close(fd)
        else:
            since = _read(self.path).get("started")
            raise StudyError(
                f"study {self.folder.name} is already running"
                + (f" (since {since})" if since else "")
                + "; wait for it to end, or cancel it"
            )
        (self.folder / CANCEL_FILE).unlink(missing_ok=True)
        self._write(None)
        return self

    def _write(self, progress: dict | None) -> None:
        # Rewritten in place: the lock is on this file, so it is never replaced.
        assert self._fd is not None  # only called while the lock is held
        text = yaml.safe_dump({"started": self._started, "progress": progress}, sort_keys=False)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.ftruncate(self._fd, 0)
        os.write(self._fd, text.encode())

    def report(self, progress) -> None:
        """Record a progress report; stop the run if it has been cancelled.

        Parameters
        ----------
        progress : aquakin.progress.Progress
            The analysis' report.

        Raises
        ------
        aquakin.Cancelled
            If ``cancel`` was created since the run started.
        """
        from aquakin.progress import Cancelled

        if (self.folder / CANCEL_FILE).exists():
            raise Cancelled()
        self._write(
            {
                "stage": progress.stage,
                "done": progress.done,
                "total": progress.total,
                "fraction": progress.fraction,
                "message": progress.message,
            }
        )

    def __exit__(self, *exc) -> None:
        fd, self._fd = self._fd, None
        assert fd is not None  # entered, or __exit__ would not run
        (self.folder / CANCEL_FILE).unlink(missing_ok=True)
        if os.name == "nt":  # an open file cannot be removed there
            _unlock(fd)
            os.close(fd)
            self.path.unlink(missing_ok=True)
            return
        # Removed before it is released, so nobody can lock it in between.
        self.path.unlink(missing_ok=True)
        _unlock(fd)
        os.close(fd)
