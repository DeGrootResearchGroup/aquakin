"""Starting, and discarding, a child process the interface talks to over its
standard streams.

The worker (:mod:`aquakin_ui.worker`) and each study run (:mod:`aquakin_ui.jobs`)
are such processes: text lines in on standard input, out on standard output
(read on a thread of its own), standard error to a log file of their own so two
never share one. :func:`start_piped` starts one and :meth:`Piped.close` is the
one way to be done with it, whatever ended it.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

__all__ = ["READER_TIMEOUT", "Piped", "start_piped"]

#: Seconds :meth:`Piped.close` waits, once the process is dead, for the reader
#: thread to reach the end of standard output.
READER_TIMEOUT = 5.0


@dataclass(frozen=True)
class Piped:
    """A started child process, its two pipes and the thread reading it.

    Attributes
    ----------
    proc : subprocess.Popen
        The process.
    stdin : IO of str
        Its standard input, in text mode, line buffered.
    stdout : IO of str
        Its standard output, in text mode. Only the reader thread reads it.
    log_path : pathlib.Path
        The file its standard error goes to. The caller removes it.
    reader : threading.Thread
        The daemon thread reading ``stdout``; it closes ``stdout`` when it ends.
    """

    proc: subprocess.Popen[str]
    stdin: IO[str]
    stdout: IO[str]
    log_path: Path
    reader: threading.Thread

    def close(self, grace: float = 0.0) -> None:
        """Discard the process: end it, reap it and close both pipes.

        Closes standard input, gives the process ``grace`` seconds to exit on
        its own, kills it if it has not, and waits for it. Only then, with the
        process dead, is the reader thread joined (for at most
        :data:`READER_TIMEOUT` seconds) and standard output closed: closing a
        pipe another thread is blocked reading could block or raise. Safe to
        call more than once, and from the reader thread itself, which then is
        past its last read.

        If the reader has not reached the end of standard output by then, the
        pipe is still held open by someone else (a process the child started
        and left running); ``close`` returns without closing ``stdout`` and the
        reader closes it when that end finally comes.

        Parameters
        ----------
        grace : float, optional
            Seconds the process may take to exit after its input closes before
            it is killed (default none: killed at once if still running).
        """
        # Flushing a line buffered pipe can only fail if the process has gone;
        # the file is closed either way.
        with contextlib.suppress(OSError):
            self.stdin.close()
        if grace > 0:
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=grace)
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        if threading.current_thread() is not self.reader:
            self.reader.join(READER_TIMEOUT)
            if self.reader.is_alive():
                return
        self.stdout.close()


def start_piped(command: list[str], log_prefix: str, read: Callable[[Piped], None]) -> Piped:
    """Start ``command`` with piped standard streams, a log and a reader thread.

    The process runs in a new session (a new process group on Windows), out of
    reach of the terminal's Ctrl-C, so the interface decides when it stops.

    Parameters
    ----------
    command : list of str
        The command line.
    log_prefix : str
        The start of the log file's name (in the temporary folder).
    read : callable
        Run on a daemon thread with the returned :class:`Piped`; reads its
        ``stdout`` to the end. ``stdout`` is closed when it returns or raises.

    Returns
    -------
    Piped
        The process, its pipes, its log file and its reader thread. Discard it
        with :meth:`Piped.close`.
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
    stdout = proc.stdout

    def pump() -> None:
        try:
            read(piped)
        finally:
            stdout.close()

    piped = Piped(proc, proc.stdin, stdout, Path(name), threading.Thread(target=pump, daemon=True))
    piped.reader.start()
    return piped
