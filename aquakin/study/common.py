"""What every part of a study shares: the names of its folder and files, its
kinds and states, :class:`StudyError`, and the time grids of ``study.yaml``.

The file names are part of the stored format: a project written by an earlier
aquakin must still read, so they do not change.
"""

from __future__ import annotations

import datetime as _dt

from pydantic import BaseModel, ConfigDict, field_validator

from aquakin.integrate.calibration.saved import COVARIANCE_FILE as COVARIANCE_FILE
from aquakin.integrate.calibration.saved import FIT_FILE as FIT_FILE
from aquakin.integrate.calibration.saved import PARAMETERS_FILE as PARAMETERS_FILE

#: The folder of a project that holds its studies, one folder each.
STUDIES_DIR = "studies"
STUDY_FILE = "study.yaml"
#: The spec file a study follows unless its ``flowsheet:`` names another (and the
#: name of the spec inside a frozen study's snapshot).
SPEC_FILE = "flowsheet.yaml"
RUN_FILE = "run.yaml"
FROZEN_FILE = "frozen.yaml"
RESULTS = "results"
#: Where a run writes; it becomes ``results`` only when the run succeeds.
PARTIAL = "results.partial"
#: A run that failed or was cancelled while earlier results stood: they are kept.
ATTEMPT_FILE = "last_attempt.yaml"
#: Present while a run is in progress, locked by it (:mod:`aquakin.study.running`):
#: when it started and its latest progress.
RUNNING_FILE = "running.yaml"
#: Created to ask the run in progress to stop at its next checkpoint.
CANCEL_FILE = "cancel"
SNAPSHOT = "snapshot"

# Result files, inside ``results/``. Their names are part of the stored format:
# a project written by an earlier aquakin must still read. A calibrate study's
# fit files (PARAMETERS_FILE, FIT_FILE, COVARIANCE_FILE) are those of a saved
# fit, imported above from aquakin.integrate.calibration.saved and re-exported
# here, where the reader and the stored-names test take them.
FINAL_STATE_FILE = "final_state.yaml"
# simulate at steady state
STREAMS_STEADY_FILE = "streams.yaml"
MEASURED_STEADY_FILE = "measured.yaml"
# simulate in time
STREAMS_FILE = "streams.csv"
MEASURED_FILE = "measured.csv"
# fisher
FISHER_FILE = "fisher.csv"
FISHER_SUMMARY_FILE = "summary.yaml"
# calibrate
FITTED_FILE = "fitted.csv"
# profile
PROFILE_FILE = "profile.csv"
PROFILE_SUMMARY_FILE = "profile.yaml"
# rank
RANKING_FILE = "ranking.csv"
# posterior: the draws of every chain, and their summary and diagnostics.
DRAWS_FILE = "draws.csv"
POSTERIOR_FILE = "posterior.yaml"
# design: the optimized experiments, the criterion and the standard errors.
DESIGN_FILE = "design.yaml"
#: Inside a frozen posterior study's ``snapshot/``: the frozen copy of its fit.
FIT_SNAPSHOT = "fit"

#: The kinds of study. Each has a runner (``_RUNNERS``), required fields
#: (``_REQUIRED``), a form entry and a result reader (:mod:`aquakin.study.form`).
KINDS = ("simulate", "fisher", "calibrate", "profile", "rank", "posterior", "design")
#: Every state :func:`check_study` can report (see :class:`StudyStatus`).
STATES = ("not run", "fresh", "stale", "damaged", "failed", "cancelled")
#: The states of a study whose last run finished, so it has results to show or replace.
RESULT_STATES = ("fresh", "stale", "damaged")


def _now() -> str:
    """The time now, UTC, to the second (ISO 8601), as run records keep it."""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


class StudyError(ValueError):
    """A study that cannot be read, run, frozen or unfrozen as asked."""


_STRICT = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _to_float(v):
    return float(v) if isinstance(v, int) and not isinstance(v, bool) else v


class _Grid(BaseModel):
    """Evenly spaced times: ``num`` points from ``start`` to ``stop``."""

    model_config = _STRICT
    start: float
    stop: float
    num: int

    _floats = field_validator("start", "stop", mode="before")(_to_float)


def _times(value) -> list[float]:
    if isinstance(value, _Grid):
        import numpy as np

        return [float(x) for x in np.linspace(value.start, value.stop, value.num)]
    return [float(x) for x in value]
