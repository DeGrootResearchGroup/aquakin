---
paths:
  - "aquakin/study/**"
  - "tests/integration/test_study.py"
  - "tests/unit/test_study_files.py"
---

# Rules — `aquakin/study/` (studies)

A study is one analysis of a flowsheet project, saved in the project with its
results. The design and the owner's decisions are in
`.claude/notes/studies-design.md`; this file is how the code realizes them.

## Files of a study (`<project>/studies/<name>/`)

| file | written by | holds |
|---|---|---|
| `study.yaml` | the user / the editor, **never rewritten by the runner** | `study:` inputs, validated strictly (`_Study`, Pydantic, `allow_inf_nan=False`) |
| `run.yaml` | `run_study` | the run record: times, aquakin version, `spec_fingerprint`, `spec_inputs`, the study's own fingerprint (`_study_fingerprint`: settings + any file they read), outcome / error, `results: {path: sha256}` |
| `results/` | `run_study` | result files (`files.py`), every value keyed by name |
| `frozen.yaml` | `freeze_study` | its presence means frozen; the snapshot's fingerprint and `spec_inputs` at freeze time (never recomputed: a spec from an older aquakin may not read through the current catalogue) |
| `snapshot/` | `freeze_study` | `flowsheet.yaml`, `models/` (every model **flattened**: its `extends` chain merged in, shipped models included as `copy_of: shipped`), `data/` |

## Rules

- **Results are keyed by name.** Parameters as `"<model>.<param>"`, plant
  states as `{unit: {state name: value}}` (`Plant.named_state`, unit
  `state_names()`), trajectories as CSV columns `<stream>.<species|Q|T|channel>`,
  matrices with named rows and columns. Floats are written with `repr`, so files
  read back bit for bit. Never store a vector by position.
- **Staleness** (`check_study`): `not run` / `failed` (`run.yaml` outcome) /
  `damaged` (a result file missing or with another sha256) / `stale` (study
  inputs changed; live: `spec_inputs` differ, with the changed key named;
  frozen: the run's fingerprint differs from the snapshot's) / `fresh`. A
  different aquakin version is a **note**, never stale (owner's decision).
- **Replacing results needs `replace=True`**; the message points at
  `copy_study`. **Unfreezing deletes the results and needs
  `discard_results=True`** (owner's decision: results computed on the snapshot
  do not describe the live plant). The editor must say so before doing either
  and offer a copy.
- A flattened model has no `extends`, so `composition_base` finds its shipped
  base by an unambiguous species match (`_base_by_species`); keep that working
  when changing `aquakin/utils/composition.py`, or measured channels (TN, COD)
  break in frozen studies.
- Kinds: `simulate` (no `time` → steady state, results `streams.yaml` /
  `measured.yaml`; with `time` → `streams.csv` / `measured.csv`; always
  `final_state.yaml` and `parameters.yaml`) and `fisher` (`fisher.csv`,
  `summary.yaml` with values, transforms, std (null when singular) and the four
  design criteria, `parameters.yaml`). A new kind adds a `_run_<kind>`, its
  required fields in `_check_kind`, and a test in
  `tests/integration/test_study.py`.
- Not yet: progress and cancel hooks, and the editor's job worker (steps 3–4
  of the design note).
