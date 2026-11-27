"""Studies: analyses of a flowsheet project saved in the project with their results.

A study (:mod:`aquakin.study.study`) is a folder under ``<project>/studies/``
holding what to run, the record of the last run and its results
(:mod:`aquakin.utils.result_files`, every value keyed by name). It is either
live (following the project's flowsheet, stale when that changes) or frozen on
a snapshot of its plant. A calibration, profile or Fisher study may study a
batch reactor in place of the flowsheet (:mod:`aquakin.study.reactor`).
``aquakin study run|check|freeze|unfreeze|copy`` is the command line.
"""

from aquakin.utils.result_files import (
    read_matrix,
    read_named_values,
    read_table,
    read_trajectory,
    write_matrix,
    write_named_values,
    write_table,
    write_trajectory,
)
from aquakin.study.form import (
    UNREADABLE,
    create_study,
    list_studies,
    plant_choices,
    read_study,
    simulation_study,
    study_form,
    study_results,
    update_study,
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
    "UNREADABLE",
    "StudyError",
    "StudyStatus",
    "check_study",
    "copy_study",
    "create_study",
    "freeze_study",
    "list_studies",
    "load_study",
    "plant_choices",
    "read_matrix",
    "read_named_values",
    "read_study",
    "read_table",
    "read_trajectory",
    "run_study",
    "simulation_study",
    "study_form",
    "study_results",
    "unfreeze_study",
    "update_study",
    "write_matrix",
    "write_named_values",
    "write_table",
    "write_trajectory",
]
