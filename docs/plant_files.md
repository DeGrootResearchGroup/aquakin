# Flowsheet files, studies and the editor

A plant can be saved as a flowsheet file, analyzed in *studies* that keep their
results beside it, and drawn and edited in a browser. The examples save the
BSM1 plant from [Plant-wide simulation](plants.md):

```python
import aquakin
from aquakin.plant.bsm import build_bsm1

model = aquakin.load_model("asm1")
bsm1 = build_bsm1(model)
bsm1_feed = {"SI": 30.0, "SS": 69.5, "XI": 51.2, "XS": 202.32, "XB_H": 28.17,
             "SNH": 31.56, "SND": 6.95, "XND": 10.59, "SALK": 7.0}
bsm1.add_influent("feed", model.influent(bsm1_feed, Q=18446.0))
```

## Saving a plant as a file

A plant can be written as a YAML (or JSON) file and rebuilt from it:

```python
bsm1.to_spec("flowsheet.yaml")
rebuilt = aquakin.Plant.from_spec("flowsheet.yaml")
```

The file lists every unit by its name, with an optional display `label` and all
of its settings (defaults included, so the file means the same plant whatever
the defaults of a later aquakin), the connections as `unit.port -> unit.port`,
the influents, named streams and recycle options, and the version of aquakin
that wrote it:

```yaml
format: 2
aquakin: 0.2.0
name: BSM1
models:
  asm1: {source: shipped, sha256: ...}
units:
  feed: {kind: influent, model: asm1, Q: 18446.0, composition: {SI: 30.0, SS: 69.5, ...}}
  tank3: {kind: cstr, label: Aerobic zone 1, model: asm1, volume: 1333.0, ...}
connections:
  - tank2.out -> tank3.inlet
```

Every model a unit uses is listed under `models:`, either a shipped model or a
model YAML file beside the flowsheet. A model file can `extends:` a shipped
model, so a variant is a few lines: BSM2's ASM1, with its parameters referenced
to 15 °C, is a `models/asm1_bsm2.yaml` that names `extends: asm1` and lists the
parameters it changes ([Inheriting a model](model_format.md#inheriting-a-model-extends)).
Each model carries a fingerprint of its content, and reading a flowsheet whose
model has changed since it was written warns.

A time-varying influent is kept as a table file beside the flowsheet (columns
`t`, `Q`, optionally `T`, and species; see `aquakin.plant.read_influent_table`).
`to_spec(path)` writes it to `data/<influent>.csv` unless it was already read
from a table inside the flowsheet's folder.

An SBR's phases are a list with one mapping per phase, in `SBRPhase`'s own
field names, and its settling model a mapping that names its `kind`
(`interface` for `InterfaceSettling`, `layered` for `LayeredSettling`) and that
model's arguments. A field left out takes its default (`mixed: null` lets the
phase decide from `feed` and `kla`), and `to_spec` writes every field. The
`feed` mode is required (a flowsheet written before SBRs had one is refused,
naming the three modes), and the holding volume's settings go with
`feed: storage`:

```yaml
  sbr:
    kind: sbr
    model: asm1
    phases:
      - {name: fill, duration: 0.1, feed: true}
      - {name: react, duration: 0.3, kla: 240.0}
      - {name: settle, duration: 0.1, settle: true}
      - {name: decant, duration: 0.1, decant: true}
    settling: {kind: interface, v_settle: 400.0, area: 200.0}
    full_volume: 1000.0
    feed: storage
    feed_flow: 5000.0
    storage_volume: 600.0
    storage_initial_fraction: 0.5
    decant_flow: 5000.0
```

Reading a file checks it strictly: a misspelled setting (a phase's too, named by
its entry), a value of the wrong type or a model that is not listed is an error
that names where it is. Anything a file cannot express yet (array-valued
settings such as initial concentrations, a heat-balance temperature model)
raises a `SpecError` that names it rather than being left out.

`aquakin.plant.spec_fingerprint(spec)` gives a fingerprint of the plant a file
builds: its settings, wiring and the content of its models and influent tables,
but not its layout, labels or where its files sit. Record it beside a result to
tell later whether the result still matches the flowsheet.

## The project folder

A flowsheet and everything it reads sit in one folder, the *project*, with the
studies of it beside them:

```text
my_project/
├── flowsheet.yaml          the plant
├── models/                 model files the flowsheet uses (shipped models stay in aquakin)
│   └── asm1_bsm2.yaml
├── data/                   influent tables
│   └── feed.csv
└── studies/
    └── effluent/           one study (below)
        ├── study.yaml      what to run
        ├── data.csv        observations, for a fit
        ├── run.yaml        the record of the last run
        ├── results/        its results
        ├── frozen.yaml     present while the study is frozen
        └── snapshot/       the frozen copy of the plant: flowsheet, models, data
```

`to_spec(path)` and the editor's **Save** copy a model or an influent table from
outside the folder into `models/` and `data/`, so the folder can be moved or
shared whole.

## Studies

A *study* is one analysis of a flowsheet saved in the project with its results.
Each study is a folder under `studies/` with a `study.yaml` saying what to run:

```yaml
# studies/effluent/study.yaml
study:
  kind: simulate
  parameters: {asm1.muH: 5.0}          # changed from the flowsheet's
  initial_state: steady                # or default, or a named-state file
  time: {t_span: [0, 14], t_eval: {start: 0, stop: 14, num: 337}}
  streams: [effluent]                  # saved whole: flow, temperature, species
  observables:
    - stream: effluent
      channels: [COD, TKN, NOx, TN, TSS]
```

```bash
aquakin study run studies/effluent      # writes results/ and run.yaml
aquakin study check studies/effluent    # fresh, stale (and why), damaged or failed
```

or `aquakin.study.run_study(path)` / `check_study(path)` from Python (the
commands are listed in [The command line](#the-command-line)). Every
result is keyed by name (parameters by `model.param`, states by unit and state
name, trajectories as CSV columns such as `effluent.SNH`), so results stay
readable after the flowsheet is reorganized. A channel is a species, a
laboratory measurement (`COD`, `TKN`, `NOx`, `TN`, `TP`, `TSS`, `Q`, `T`) or a
weighted sum of your own; see
[A channel of your own](plant_calibration.md#a-channel-of-your-own).

Besides `simulate`, a study can be a `fisher` information of a sensor design, a
`calibrate` fit to observations, a `profile` likelihood of one parameter, or a
`rank` of candidate sensor sets (the methods are described in
[Designing experiments](experiment_design.md)).
A fit reads its observations from a CSV table in the study folder, a `t` column
and one `<stream>.<channel>` column per measured channel. Because it fits the
likelihood of the data, it needs each channel's measurement error: `sigma`, one
value for all channels or one per channel, or `estimate_sigma: true` to
estimate it from the fit:

```yaml
study:
  kind: calibrate
  free: [asm1.muH, asm1.KS]
  observables:
    - {stream: effluent, channels: [SNH, TSS]}
  data: data.csv          # columns t, effluent.SNH, effluent.TSS
  sigma: [0.2, 2.0]       # each channel's measurement error
  laplace: true
```

A study follows the flowsheet: once a setting, a connection, a model or an
influent table it was run on changes, `check` reports it **stale** and names
what changed. A newer aquakin is noted but does not make a result stale; read
the changelog to decide whether to run it again. Running a study that has
results replaces them only when asked (`--replace`); `aquakin study copy` keeps
a copy first. The new results take the old ones' place only when the run
succeeds: a run that fails or is cancelled leaves the earlier results as they
were, and `check` notes the attempt.

`aquakin study freeze` instead pins a study to a snapshot of its plant, taken
inside the study folder with every model flattened (shipped models too) and
every influent table copied, so neither an edit to the project nor an aquakin
upgrade can change what it runs. `aquakin study unfreeze --discard-results`
makes it follow the flowsheet again and deletes its results, which were
computed on the snapshot.

From Python, `aquakin.study.create_study(project, name, settings)` and
`update_study(path, settings)` write a `study.yaml`, checking the settings as a
run would read them; `plant_choices(spec)` lists what a study of that flowsheet
may name, and `study_results(path)` reads a study's results back.

## The flowsheet editor

```bash
pip install "aquakin[ui]"
aquakin-ui flowsheet.yaml
```

([Installing the editor without Python](installing_the_editor.md) covers a
computer without Python: one download or one pasted line.)

opens a flowsheet editor in your browser on the flowsheet saved above (or an
empty one, without a file), served on this machine only. Drag
units from the palette, wire their ports, and edit each unit in a form generated
from its own documentation; aquakin builds each unit and then the whole plant as
you edit and lists any errors.

- **Open** and **Save** read and write the files described above. Save writes
  exactly what `plant.to_spec(path)` would, copying a model (with every file it
  `extends:`) or influent table from outside the project folder into it. Save
  refuses to overwrite a flowsheet that changed on disk since the window opened
  it, so two windows cannot silently undo each other's work.
- Opening another file while the flowsheet has unsaved edits asks first (save
  them, or discard them). **Undo** takes back the last edits: a removed unit or
  wire, a move, a new unit or wire, or an applied form. Delete removes the
  selected units and wires (Backspace does not).
- Edits typed into a unit's form are applied when you select another unit, and
  a value that cannot be read is named under its field.
- A chemical dosing unit's reagent is set in its form: choose the model it is
  written in first, then add each species it contains with its concentration.
- A sequencing batch reactor's phases are a table in its form: add a phase,
  move it up or down, or remove it. Its settling model is a choice of kind,
  after which that model's arguments appear.

**Simulate** saves the flowsheet, creates a study called `simulation` the first
time (`simulation_<file name>` for a file not named `flowsheet.yaml`, so each
flowsheet in a folder has its own), runs the plant to steady state, and shows a
table of every stream that leaves the plant: its flow, COD, TKN, NOx, TN and TSS,
with every species under *Flow and species* (temperatures in °C). To run in time
instead, or to save other streams, **Edit** the `simulation` study; a study that
runs in time shows a chart per channel of the stream you pick. From Python,
`aquakin.study.simulation_study(spec)` gives those settings.

The **Studies** tab lists the project's studies and their status (fresh, stale
and why, not run, failed, cancelled). From it you can run a study and watch its
progress or cancel it; freeze, unfreeze or copy it; and open **Results**: a
simulation's streams, a Fisher study's standard deviations, design criteria and
information matrix, a ranking of sensor sets, a calibration's fitted parameters
with the data against the fit, or a profile's curve and interval. Running a
study first saves the flowsheet's unsaved edits, so it runs what you see (a
frozen study runs its snapshot). Replacing results or unfreezing asks first and
offers to keep a copy. Tick **Don't ask again** to replace results without the
question from then on; the **Ask before replacing results** switch at the top of
the tab brings it back (the choice is kept in `aquakin/ui.json` in your
configuration folder). Unfreezing always asks.

**New study** sets one up in a dialog: pick its kind, and the fields that kind
uses appear, with the plant's own streams, parameters and measurable channels to
choose from; **Edit** opens a study's settings in the same dialog. The dialog
writes the same `study.yaml` you would write by hand, and a data table or saved
state chosen from elsewhere is copied into the study folder. A study set up in
the editor follows the flowsheet file that is open.

## The command line

Installing aquakin installs two commands:

| Command | What it does |
|---|---|
| `aquakin study run <study> [--replace]` | runs a study and writes its results; a study that has results is run again only with `--replace` |
| `aquakin study run <study> --json` | the same, reporting progress as JSON lines for a program that drives it; a line `cancel` on standard input, or closing it, stops the run at its next checkpoint |
| `aquakin study check <study>` | prints whether the results are fresh, stale (and why), not run, failed or cancelled; exits with 0 when fresh or not run, 1 otherwise |
| `aquakin study freeze <study>` | pins the study to a snapshot of its plant |
| `aquakin study unfreeze <study> --discard-results` | makes it follow the flowsheet again and deletes its results |
| `aquakin study copy <study> <name>` | copies the study, results included, beside it under `<name>`, and prints the new folder |
| `aquakin catalog` | prints every unit kind a flowsheet file may use, with its settings and ports, as JSON |
| `aquakin-ui [<flowsheet>] [--port <n>] [--no-browser]` | opens the editor; `--port` serves it on a fixed port (by default a free one), and `--no-browser` prints the address instead of opening a browser |

`<study>` is the study's folder, such as `studies/effluent`. A command that
fails exits with 2 and prints the reason.
