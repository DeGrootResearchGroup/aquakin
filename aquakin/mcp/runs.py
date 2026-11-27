"""Study runs started by an assistant: in the background, with progress and cancel.

A study can run for minutes or hours, longer than an assistant's app waits for
one tool call. So ``run_study`` starts the run on a thread of its own and
returns at once; the assistant asks ``study_status`` how it is going and can
``cancel_study``. Each run is :func:`aquakin.study.run_study`, the function the
``aquakin study run`` command calls, writing the same files: a study run here
shows in the editor like any other.
"""

from __future__ import annotations

import threading
from pathlib import Path

from aquakin.plant.errors import error_reply
from aquakin.progress import Cancelled


class _Run:
    """One study running on its own thread."""

    def __init__(self, folder: Path, replace: bool):
        self.folder = folder
        self.replace = replace
        self.progress: dict | None = None  # the latest progress report
        self.outcome: dict | None = None  # how it ended, once it has
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _report(self, p) -> None:
        # Called by the analysis at each checkpoint: the place a cancel stops it.
        if self._cancel.is_set():
            raise Cancelled()
        self.progress = {
            "stage": p.stage,
            "done": p.done,
            "total": p.total,
            "fraction": p.fraction,
            "message": p.message,
        }

    def _run(self) -> None:
        from aquakin.study import run_study

        try:
            run_study(self.folder, replace=self.replace, progress=self._report)
            self.outcome = {"outcome": "finished"}
        except Cancelled:
            self.outcome = {"outcome": "cancelled"}
        except Exception as exc:  # the run's thread: every failure is its outcome
            self.outcome = {"outcome": "failed", **error_reply(exc)}

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the run to end; whether it has."""
        self._thread.join(timeout)
        return not self.running


class StudyRuns:
    """The runs one server has started, by study folder."""

    def __init__(self):
        self._runs: dict[Path, _Run] = {}
        self._lock = threading.Lock()

    def start(self, path, *, replace: bool = False) -> _Run:
        """Start running a study; refuse if this server is already running it.

        Raises
        ------
        ValueError
            If the study is already running here.
        """
        folder = Path(path).resolve()
        with self._lock:
            current = self._runs.get(folder)
            if current is not None and current.running:
                raise ValueError(
                    f"study {folder.name} is already running; ask study_status how it is "
                    f"going, or cancel_study to stop it"
                )
            self._runs[folder] = _Run(folder, replace)
            return self._runs[folder]

    def get(self, path) -> _Run | None:
        """The latest run of a study started here, or ``None``."""
        return self._runs.get(Path(path).resolve())
