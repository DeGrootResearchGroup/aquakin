"""Progress reports from long analyses, and cancelling them.

The plant analyses that can run for minutes (:meth:`Plant.calibrate`,
:func:`~aquakin.plant.plant_profile_likelihood`,
:func:`~aquakin.plant.plant_rank_sensor_sets`,
:func:`~aquakin.plant.plant_fisher_information`) take a ``progress=`` callable.
It is called with a :class:`Progress` at each checkpoint: every objective
evaluation of a fit, every profile point, every run of a design. To cancel
the analysis, raise :class:`Cancelled` from the callable; it propagates out of
the analysis unchanged (it is not an ``Exception``, so no failure handling on
the way catches it).

Examples
--------
>>> import threading
>>> stop = threading.Event()
>>> def progress(p):
...     print(p.stage, p.done, p.total, p.message)
...     if stop.is_set():
...         raise aquakin.Cancelled()
>>> plant.calibrate(obs, t_obs, ["asm1.muH"], progress=progress)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["Cancelled", "Progress", "report"]


@dataclass(frozen=True)
class Progress:
    """One progress report.

    Attributes
    ----------
    stage : str
        What is running: ``"fit"``, ``"profile"``, ``"design"``.
    done : int
        Steps finished in this stage (objective evaluations, profile points,
        runs).
    total : int or None
        Steps in the stage when known in advance (``None`` for a fit, whose
        number of evaluations is not).
    message : str
        A short description of the current step.
    """

    stage: str
    done: int
    total: int | None = None
    message: str = ""

    @property
    def fraction(self) -> float | None:
        """``done / total``, or ``None`` when the total is not known."""
        return None if not self.total else min(self.done / self.total, 1.0)


class Cancelled(BaseException):
    """Raised by a ``progress`` callable to stop the analysis that called it.

    A ``BaseException``, like ``KeyboardInterrupt``: the analyses turn failed
    solves into gaps or skipped starts with ``except Exception``, and a cancel
    must not be mistaken for one.
    """


def report(
    progress: Callable[[Progress], object] | None,
    stage: str,
    done: int,
    total: int | None = None,
    message: str = "",
) -> None:
    """Send one report to ``progress`` (nothing when it is ``None``)."""
    if progress is not None:
        progress(Progress(stage, done, total, message))
