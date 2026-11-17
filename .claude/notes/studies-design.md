# Studies: design proposal (2026-11-17)

Status: proposal for the owner's review; nothing here is implemented.

A **study** is one analysis of a flowsheet project, saved in the project with
its results: a simulation, a calibration, a Fisher information, a profile
likelihood or a sensor ranking. This note fixes how studies are stored, how they
know whether they are out of date, and how they run, using the owner's decisions
of 2026-11-15 and 2026-11-17.

## Decisions already made

- **Live and frozen studies.** A live study follows the flowsheet and is marked
  stale when what it simulates changes. A frozen study keeps its own snapshot of
  the plant and is never stale.
- **Re-running replaces the results**, after a warning; to keep the old results
  the user saves a copy of the study first.
- **An aquakin version change is noted, not marked stale.** Whether to re-run is
  the user's call, from the changelog.
- **Stored vectors are keyed by name**, never by position.
- **Freezing copies the shipped models** the plant uses into the snapshot, so an
  upgrade cannot change a frozen study.
- **One editor process per window** for now; a launcher opens a new window as a
  new process.

## Files

```
project/
  flowsheet.yaml
  models/  data/                 # as today
  studies/
    <study>/
      study.yaml                 # what to run (inputs), and the run record
      results/                   # what the run produced
      snapshot/                  # frozen studies only
        flowsheet.yaml  models/  data/
```

`study.yaml` has three parts:

1. **`study:`** the inputs: `kind` (`simulate`, `calibrate`, `fisher`,
   `profile`, `rank`), `mode` (`live` or `frozen`), and the kind's settings.
   - Observables are `{stream, channels}` lists. A channel is a species, a
     measurement keyword, or `{measure: name, weights: {...}, flow, temperature}`
     (the serialized `Measure`).
   - Data files (observations) are copied into `studies/<study>/data/` on save,
     like influent tables.
2. **`run:`** the record of the last run, written by the runner:
   - the time and aquakin version;
   - `spec_fingerprint` and the per-input `spec_inputs` at run time, so a stale
     note can say *which* input changed;
   - the study inputs' own fingerprint;
   - the outcome (finished, cancelled, failed with its message).
3. **`results:`** a list of result files with their sha256.

**Results** are plain files, keyed by name:
- **Parameters** are a mapping `"<model>.<param>" -> value`.
- **Stream trajectories** are CSVs with a `t` column and named channel columns.
- **Matrices** (Fisher information, covariance) are CSVs with row and column
  names.
- **Plant states** (a fitted `y0`, a steady state) are `{unit: {state name:
  value}}`.

## Staleness

Each check compares the stored `run:` record with the present:

| situation | live study | frozen study |
|---|---|---|
| `spec_fingerprint` differs | **stale**, naming the changed input (flowsheet, model, table) | not stale (built from its snapshot) |
| study inputs changed since the run | **stale** | **stale** |
| aquakin version differs | noted | noted |
| a result file is missing or edited | **results damaged** | **results damaged** |

A frozen study stores `spec_inputs` at freeze time and never recomputes them
through the current catalogue. A spec written by an older aquakin may have a
setting since renamed, and recomputing would then fail.

## Running

- **A separate job worker.** Each editor process gets a second
  `python -m aquakin serve` worker for runs, so a long run never queues the
  interactive checks or Save.
- **A job protocol.** The current protocol gives one reply per request and kills
  a request after 120 s. The job worker adds:
  - `start_study` returns a job id at once;
  - the worker then sends `progress` lines (stage, fraction, message) and one
    final `done` line;
  - `cancel` stops a job at its next checkpoint.
  - Results are written to files by the worker. Only references come back over
    the pipe.
- **Library hooks this needs** (useful outside the editor too):
  - a `progress=` callback and a cancellation check on `calibrate_plant` (per
    multistart and per optimizer iteration), `plant_profile_likelihood` (per
    grid point) and `plant_rank_sensor_sets` (per draw and run);
  - `Plant.solve` reports per output chunk where that is cheap.
- **Per window.** `aquakin-ui project/` opens one window, and a "New window"
  action launches another process. The processes share nothing, so two windows
  on the same project can each write `flowsheet.yaml`. To stop one silently
  replacing the other's work, Save refuses when the file changed since it was
  opened ("changed on disk; reload or save as").

## Prerequisites in the library

1. **State names for every unit:** a `state_names()` in the unit contract, so
   that non-species state blocks (Takács layers, the IFAS biofilm, filter
   states) can be stored by name. The CSTR and digester already have them
   through their species.
2. **A study runner in the library**, `aquakin.study.run(path)`, independent of
   the editor (the editor calls it through the job worker). It reads
   `study.yaml`, builds the plant from the flowsheet or the snapshot, runs the
   analysis and writes the results and the run record. Also an
   `aquakin study run|check <path>` command line.
3. **The progress and cancel hooks** above.

## Proposed order

1. Library: `state_names()` and the result writers (named parameters, streams,
   matrices, states).
2. Library: the `study.yaml` schema, the runner and the stale check for
   `simulate` and `fisher`; `aquakin study` on the command line.
3. Library: progress and cancel hooks; then the `calibrate`, `profile` and
   `rank` study kinds.
4. Editor: the job worker and its protocol, a Studies panel (list with status,
   run, cancel, open results), freeze and unfreeze, and Save's
   changed-on-disk guard.

## Open questions for the owner

1. **Unfreezing.** Should unfreezing a study discard its snapshot and make it
   follow the flowsheet again (results then possibly stale), or should it be
   refused?
2. **Influent tables in a frozen snapshot.** Copied like the models (proposed),
   or referenced from `data/` by hash?
3. **First study kinds.** Proposed: `simulate` (steady state or dynamic, chosen
   streams saved) and `fisher`. Are those the right first two, or is
   calibration the priority?
