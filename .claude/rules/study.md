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
- Kinds (`_RUNNERS`, each `_run_<kind>(plant, folder, study, params, y0, out,
  progress)`; required fields in `_REQUIRED`):
  - `simulate`: no `time` → steady state (`streams.yaml`, `measured.yaml`);
    with `time` → `streams.csv` / `measured.csv`; always `final_state.yaml`
    and `parameters.yaml`.
  - `fisher`: `fisher.csv`, `summary.yaml` (values, transforms, std or null
    when singular, the four design criteria), `parameters.yaml`.
  - `calibrate`: observations from `data:` (a `write_trajectory` table, one
    `<stream>.<channel>` column per channel, so channels must be named);
    `fit.yaml`, `parameters.yaml` (the fitted vector), `fitted.csv` (the
    fitted plant's channels at the data times), `covariance.csv` with
    `laplace`. Priors are on by default (`use_priors`), as in `Plant.calibrate`.
  - `profile`: the calibrate fields plus `profile` and `grid`; `profile.csv`
    (`write_table`) and `profile.yaml` (MLE, interval).
  - `rank`: `candidates` (named sensor sets) and optional `base`;
    `ranking.csv`.
  Data and initial-state files are part of the study's fingerprint.
  A new kind adds a runner, its required fields, and a test in
  `tests/integration/test_study.py`.
- **Progress and cancel** (`aquakin/progress.py`): `run_study(progress=)` passes
  the callable to the analysis; `Cancelled` (a `BaseException`) stops it and
  the run is recorded with outcome `cancelled` (state `"cancelled"`).
- **Setting a study up** (`form.py`): `study_form()` describes every `_Study`
  field (type, label, help, default) and each kind's fields and required ones
  (`_REQUIRED`); a test pins that it covers every field but `kind` and
  `flowsheet`, so a new setting needs an entry. `plant_choices(spec)` gives the
  streams (named, then every `unit.port`), parameters and per-stream channels.
  `create_study` / `update_study` validate exactly as a run reads (`_Study` +
  `_check_kind`) and copy a data or initial-state file from elsewhere into the
  study (never overwriting another file of that name); `read_study` returns the
  settings without defaults. `simulation_study(spec)` is the editor's Simulate:
  steady state, the plant's outlets (outputs nothing is connected to, by stream
  name when they have one, else every named stream) with their measurement
  keywords. `study_results(path)` reads results for a front end: every kind
  gives its file names; `simulate` also its values by stream and channel
  (`_by_stream` splits `<stream>.<channel>` columns by the study's own stream
  names, since a stream name may hold a dot), non-finite values as `None`, and
  temperature converted to °C (the files keep kelvin), and
  units built from the plant when it still builds (`_measure_units` reads the
  measurement keywords' units off the species'). The worker answers all of them.
- The editor drives studies through the job worker (`aquakin/__main__.py`'s
  `start_study` / `cancel_study`); see `.claude/rules/ui.md`.
