"""Study runs started by an assistant: each a process of its own, detached.

A study can run for minutes or hours, longer than an assistant's app waits for
one tool call or stays open. So ``run_study`` starts ``aquakin study run
<folder>`` as a process detached from this server and returns once the run
holds its study (:mod:`aquakin.study.running`). From then on everything about
the run is in the study folder: its progress, its cancel, and its record and
results when it ends. The run goes on if the server exits, and the editor and
the command line see it like any other.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aquakin.study.common import StudyError

#: Seconds a run may take to start (it imports aquakin and JAX) before
#: ``start_run`` returns without having seen it hold its study.
START_TIMEOUT = 120.0
#: Seconds between looks at a run that is starting.
_POLL = 0.1

#: The runs this process started and has not yet seen end, with the file their
#: error output goes to; each is reaped once it has ended.
_STARTED: list[tuple[subprocess.Popen, Path]] = []


def _reap() -> None:
    """Forget the runs that have ended (reaping each), removing their logs."""
    for entry in list(_STARTED):
        proc, log = entry
        if proc.poll() is not None:
            _STARTED.remove(entry)
            with contextlib.suppress(OSError):
                log.unlink()


def _detached() -> dict:
    """``Popen`` options for a process that outlives this one and its terminal."""
    if os.name == "nt":
        return {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def start_run(folder: Path, *, replace: bool = False) -> subprocess.Popen:
    """Start ``aquakin study run`` on a study, detached; return once it is under way.

    It returns when the run holds its study, when the process has already
    ended (a short run, or one that failed: its outcome is then in the study's
    record), or after :data:`START_TIMEOUT` seconds.

    Parameters
    ----------
    folder : pathlib.Path
        The study folder.
    replace : bool, optional
        Replace the study's results.

    Returns
    -------
    subprocess.Popen
        The run's process.

    Raises
    ------
    StudyError
        If the run was refused before it began (it exits with 2, and its error
        output says why).
    """
    from aquakin.study.running import run_state

    _reap()
    fd, log_name = tempfile.mkstemp(prefix="aquakin-run-", suffix=".log")
    log = Path(log_name)
    command = [sys.executable, "-m", "aquakin", "study", "run"]
    command += [*(["--replace"] if replace else []), str(folder)]
    with os.fdopen(fd, "w") as err:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=err,
            **_detached(),
        )
    _STARTED.append((proc, log))
    end = time.monotonic() + START_TIMEOUT
    # Until this run holds the study: another program's run may hold it now.
    while (
        proc.poll() is None
        and (run_state(folder) or {}).get("pid") != proc.pid
        and time.monotonic() < end
    ):
        time.sleep(_POLL)
    if proc.poll() == 2:  # refused before the run began: nothing was recorded
        reason = log.read_text().strip().splitlines()[-1:] or ["no reason given"]
        raise StudyError(f"the run of study {folder.name} did not start: {reason[0]}")
    return proc
