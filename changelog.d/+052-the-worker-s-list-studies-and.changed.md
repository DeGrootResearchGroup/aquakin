The worker's `list_studies` and `study_results` replies, and the `done` event
of `aquakin study run --json`, say whether the study has results
(`has_results`); `list_studies` also gives the spec each study builds its plant
from (`flowsheet`), and `study_results` where the study stands (`state`,
`reasons`, `notes`). The editor's results now say why they are out of date
(which input changed) rather than only that the flowsheet changed.
