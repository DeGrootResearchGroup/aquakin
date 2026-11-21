"""Starting a child process the interface talks to over its standard streams.

The worker (:mod:`aquakin_ui.worker`) and each study run (:mod:`aquakin_ui.jobs`)
are such processes: text lines in on standard input, out on standard output,
standard error to a log file of their own so two never share one.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO

__all__ = ["Piped", "start_piped"]


@dataclass(frozen=True)
class Piped:
    """A started child process and its two pipes.

    Attributes
    ----------
    proc : subprocess.Popen
        The process.
    stdin : IO of str
        Its standard input, in text mode, line buffered.
    stdout : IO of str
        Its standard output, in text mode.
    log_path : pathlib.Path
        The file its standard error goes to. The caller removes it.
    """

    proc: subprocess.Popen[str]
    stdin: IO[str]
    stdout: IO[str]
    log_path: Path


def start_piped(command: list[str], log_prefix: str) -> Piped:
    """Start ``command`` with piped standard streams and a log of its own.

    The process runs in a new session (a new process group on Windows), out of
    reach of the terminal's Ctrl-C, so the interface decides when it stops.

    Parameters
    ----------
    command : list of str
        The command line.
    log_prefix : str
        The start of the log file's name (in the temporary folder).

    Returns
    -------
    Piped
        The process, its pipes and its log file.
    """
    if sys.platform == "win32":
        creationflags, new_session = subprocess.CREATE_NEW_PROCESS_GROUP, False
    else:
        creationflags, new_session = 0, True
    fd, name = tempfile.mkstemp(prefix=log_prefix, suffix=".log")
    with os.fdopen(fd, "w") as log:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
            start_new_session=new_session,
            creationflags=creationflags,
        )
    # Both streams were asked for as pipes above, so neither is None.
    assert proc.stdin is not None and proc.stdout is not None
    return Piped(proc, proc.stdin, proc.stdout, Path(name))
