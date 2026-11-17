"""Studies: analyses of a flowsheet project saved in the project with their results.

A study (:mod:`aquakin.study.study`) is a folder under ``<project>/studies/``
holding what to run, the record of the last run and its results
(:mod:`aquakin.study.files`, every value keyed by name). It is either live
(following the project's flowsheet, stale when that changes) or frozen on a
snapshot of its plant. ``aquakin study run|check|freeze|unfreeze|copy`` is the
command line.
"""

from aquakin.study.files import (
    read_matrix,
    read_named_values,
    read_trajectory,
    write_matrix,
    write_named_values,
    write_trajectory,
)
from aquakin.study.study import (
    StudyError,
    StudyStatus,
    check_study,
    copy_study,
    freeze_study,
    load_study,
    run_study,
    unfreeze_study,
)

__all__ = [
    "StudyError",
    "StudyStatus",
    "check_study",
    "copy_study",
    "freeze_study",
    "load_study",
    "read_matrix",
    "read_named_values",
    "read_trajectory",
    "run_study",
    "unfreeze_study",
    "write_matrix",
    "write_named_values",
    "write_trajectory",
]
