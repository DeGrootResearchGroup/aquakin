---
paths:
  - "aquakin_ui/**"
  - "tests/ui/**"
  - "tests/browser/**"
  - "ui-widget/**"
---

# Rules — `aquakin_ui/` (the browser flowsheet editor, `aquakin-ui`)

A local browser flowsheet editor: palette, Vue Flow canvas, a unit form generated
from aquakin's unit catalogue, aquakin's checks, and Open/Save of flowsheet specs
(`Plant.to_spec` / `Plant.from_spec`, see `.claude/rules/plant-spec.md`). trame +
Vuetify, the same stack as aquaflux's `aquaflux-ui`. `ui` extra; entry point
`aquakin-ui [flowsheet.yaml] [--port] [--no-browser]` (`aquakin_ui.__main__:main`).

## ⚠️ The independence rule (binding, mechanically checked)

**`aquakin_ui` imports neither `aquakin` nor `jax`; `aquakin` never imports
`aquakin_ui`.** Everything about units, models and plants comes from one
long-lived `python -m aquakin serve` worker (`aquakin/__main__.py`, client
`aquakin_ui/worker.py`). Pinned by `tests/ui/test_independence.py`, which
finds every `aquakin_ui/**/*.py` itself: an AST scan of each for an `aquakin` /
`jax` import anywhere in the file (a function-level import counts, though a
fresh import would never run it), a fresh-interpreter import of every module,
and the reverse AST scan of `aquakin/`. Tests may import aquakin (to build
expected values). `tests/ui` runs in the required fast gate (the `ui` CI job).

## Structure

| module | holds | notes |
|---|---|---|
| `worker.py` | `AquakinWorker`, `WorkerError` | one process, requests serialized by a lock, replies read on a daemon thread; a fresh worker's first reply gets `START_TIMEOUT` extra (importing JAX); reply ids checked; stderr to a per-worker log file (`mkstemp`, removed on `stop`) quoted in errors |
| `catalog.py` | `Catalog`, `plain` | the worker's catalogue reply, its summaries and help made plain text (`plain`: docstring ``code``, :roles: and *emphasis* lose their markup); `ports_for` reads the catalogue's port rules (no round trip) |
| `document.py` | `Flowsheet`, `Unit`, `edge_id` | **pure**: units keyed by stable id (the plant's unit name), labels as display text; `from_spec`/`to_spec` (format 2; `from_spec` refuses a kind the catalogue lacks, which would otherwise break every redraw), `connect` (refuses with a reason), `dropped_by` (previews the connections new settings would remove) and `apply`, `remove` (also drops streams/endpoints naming the unit), `port_status`, `back_edges` (aquakin's recycles when known, else a DFS), `lanes`, `auto_layout` (wraps into bands of 7 ranks) |
| `forms.py` | `rows_for`, `read_rows` | **pure**: one row per catalogue setting (conditions and aeration expand); read-only settings and the aeration fields the form does not show are kept on read-back. **Numbers are shown exactly** (`_float_text`, the shortest round-trip text): Apply reads every field back, so a rounded display would change values nobody edited. The aeration value is read only in the mode the form was drawn with, and a unit's conditions are kept when its model could not be described (no condition rows). Each row carries a `suffix`, the unit its help names (`unit_in`: "(m³)", "(g/m³, i.e. mg/L)", ", m³/d"; help writes flows and times per day and they are shown on the plant's clock, m³/h on an hourly plant), a condition's model unit (T in K), kLa's 1/clock, the DO set-point's oxygen unit. An unreadable field raises `FieldError` with its row `key` (shown under the field from `sel_errors`). `suggest_species` gives settling settings the model's X species when their ASM1 default (or empty one) does not fit; new units and model changes take it. Lists no setting or type of its own (the settling settings are named by their role) |
| `icons.py` | `ICONS`, `COLOURS`, `CATEGORY_LABELS`, `SVG`, `icon_name` | the only per-kind presentation table; an unlisted kind gets the tank icon |
| `study_form.py` | `fields_for`, `draft_for`, `study_from`, `missing` | **pure**: the study dialog's draft from the worker's `study_form` (numbers kept as text, an even time grid as start / stop / count) and back, empty fields and defaults left out; `missing` names a kind's required fields by label. Lists no setting of its own |
| `results.py` | `fmt`, `steady_table`, `channels_of`, `line_chart`, `xy_chart`, `analysis_view`, `ANALYSIS_VIEWS`, `SIMULATE` | **pure**: the results view's tables (a column per stream, a row per channel with its unit) and SVG charts (one channel each, so never two y-axes; the dataviz palette's slot-1 blue, `#3987e5` in dark mode, set in `PAGE_CSS`; a CSS-hover crosshair and a native `<title>` tooltip per sampled time, at most `MAX_HOVER` of them, the line thinned to `MAX_POINTS`). `xy_chart` is the core: a line, data points (slot-2 orange, with a fit/data legend when both) and a labelled reference hairline (a profile's threshold). `analysis_view` dispatches through `ANALYSIS_VIEWS` (one `_<kind>_view` per kind but `SIMULATE`, which the editor draws itself) and raises on a kind it lacks |
| `settings.py` | `Settings`, `DEFAULTS`, `settings_path` | the user's preferences, a JSON file in the configuration folder (`AQUAKIN_UI_CONFIG` overrides; `tests/ui/conftest.py` points it at a temporary file for every test, so tests never touch the user's); unknown keys and wrong types are ignored on read; a failed write keeps the value for the session |
| `jobs.py` | `StudyRun` | one study run = one `python -m aquakin study run --json` process: JSON progress events on stdout, `cancel` on stdin (killed after `CANCEL_GRACE` if no checkpoint comes); its own temporary log |
| `app.py` | `Editor` | the page; turns finished canvas edits into document changes; the Studies tab |
| `widget/` | `Flowsheet` (trame element), `module.py`, `serve/` | the built Vue Flow bundle, committed; source in `ui-widget/` |

Icons (`icons/*.svg`) are original line icons (see `icons/README.md`): 24-unit
viewBox, `currentColor` strokes, one shared vocabulary. Keep new ones to it.

## The worker protocol (pinned by `tests/ui/test_protocol.py`)

The two sides share no import, so what they must agree on is pinned by a test
that imports both. Keep it that way rather than copying knowledge across:

- **Every command name the editor sends** (`worker.request("...")`,
  `_worker_study("...")`, always a literal) is one of `_handlers()` in
  `aquakin/__main__.py`. A new command is added there first.
- **A refusal is `{"error": message, "type": class name}`**
  (`aquakin.plant.errors.error_reply`; also `check_unit` / `check_spec` results
  and the run's `error` event). `WorkerError` carries the message as its text and
  `.type`. Show the message as it stands; never parse a class name out of text.
- **Study state and paths come from the worker**: `list_studies` gives each
  study's `path`, `state`, `has_results` and `flowsheet` (the spec it builds
  from); `study_results` and the run's `done` event give `has_results` and the
  reasons. Decide "has results" from `has_results`, never from a state's name,
  and never rebuild `studies/<name>/study.yaml` or `../../flowsheet.yaml` here:
  a new study's `flowsheet:` is sent as the spec's absolute path and aquakin
  stores it relative. `STATE_COLOURS` (app.py) is the one table of state names
  on this side, pinned to aquakin's `STATES`.
- **Every study kind has a view** (`results.SIMULATE` + `ANALYSIS_VIEWS` equal
  aquakin's `KINDS`).

## Sync rules (each was a bug found in the browser)

- **Python owns the flowsheet.** The canvas draws `nodes`/`edges` and emits only
  finished edits (`connect`, `move`, `remove`, `dropped`, `select`); dragging,
  panning and wire drawing stay in the browser.
- **Apply Python's updates as a diff** (`syncNodes`/`syncEdges` in
  `ui-widget/src/FlowsheetCanvas.vue`): replacing Vue Flow's lists rebuilds every
  node (losing its measured size) and re-validates every edge, which drops them.
- **Wire validation sits on the handles** (`UnitNode.vue`), never on Vue Flow's
  canvas-wide `is-valid-connection`, which it also applies to loaded edges.
- **No echoes:** removals and selection changes the canvas makes while applying
  Python's update are not reported back (only ids Python still has; the
  selection reference is reset after each sync). Python marks its selected unit.
- **The form is edited in the browser and sent whole** on Apply (or when the
  aeration mode / model changes the rows): a per-keystroke round trip lost clicks
  and overwrote text. Help sits in a fixed panel, not under each field (a hint
  collapsing on blur moved Apply between press and release).
- **aquakin's checks run in the background** (`Editor.schedule_checks`, after
  `CHECK_DELAY` of quiet, worker calls in an executor), each unit and the plant
  cached by its spec, so the canvas never waits; `checking` shows meanwhile and
  "None" is claimed only after a check finished. A move only republishes. The
  document is read on the loop thread (`_unit_check`, `_plant_request`
  snapshot each request); only the worker call (`_ask_unit`, `_ask_plant`) runs
  in the executor, so an edit made during a check cannot be half-read.
- **Open and Save go through aquakin** (`read_spec` and `write_spec` worker
  commands): Open reads YAML 1.2 like `Plant.from_spec`; Save is
  `aquakin.plant.spec.write_spec` (settings filled in, version and hashes
  recorded, outside model and table files copied into `models/` and `data/`,
  atomic replace), and the document is reloaded from what was written (unless
  it was edited while saving). Both run off the page's thread (`_off_page`, a
  `busy` state disables the buttons meanwhile; without a served page, at start-up
  and in tests, they run inline). Never write the spec file from `aquakin_ui`
  directly. `_check_unit` / `_check_plant` are the synchronous versions for
  tests and scripts; a page handler must not call them.
- **Save refuses to overwrite work** (`_save_refusal`): each window is its own
  document, so two may hold one file. Save refuses a file whose sha256 changed
  since this window opened (or last saved) it, and any other existing file.
- **Studies** (`Editor.refresh_studies` / `run_study` / `cancel_run` /
  `freeze_study` / `unfreeze_study` / `copy_study`): listed through the worker
  (`list_studies`, with `check_study`'s state); freeze / unfreeze / copy are
  worker commands; a run is a `StudyRun` process, never a worker request (the
  worker kills a request after `TIMEOUT`, and a run must not hold up checks or
  Save). Run events arrive on a reader thread and are applied on the page's
  loop (`call_soon_threadsafe`). **Replacing results and unfreezing ask first**
  (`_ask` / `answer_study`) and offer "Copy first", which copies the study with
  its results, then goes ahead (owner's decision). Replacing on a run carries a
  "Don't ask again" box and the Studies tab an "Ask before replacing results"
  switch (owner's decision); both go through `Editor.set_ask_replace`, which
  keeps the `ask_before_replacing` preference (the switch calls it on change;
  nothing watches the state, so code must call the method, not set the state).
  Unfreezing always asks.
- **The study dialog** (`Editor.new_study` / `edit_study` / `set_study_kind` /
  `save_study_dialog`): fields come from the worker's `study_form`, choices from
  `plant_choices` (the drawn ports when the plant does not build); switching kind
  keeps the fields both kinds have. The draft is edited in the browser and sent
  whole on Create / Save, and written by the worker (`create_study` /
  `update_study`) off the page's thread; errors show beside the buttons, not at
  the foot of the scrolled form. **Values inside the draft are nested, and the
  page does not redraw on a nested change**: every widget bound into it uses
  `_bound` (which reassigns `study_draft` after each change) and every row
  add / remove ends with `_REDRAW`, or rows do not appear and a stream's
  channels stay empty. Each field heading carries the `<input id>-label` id
  Vuetify's `aria-labelledby` points at.
- **Simulate** (`Editor.simulate`): saves first when the document differs from
  what was last opened or saved (`_saved`, a JSON snapshot; a move counts),
  then, off the page's thread, creates the spec's Simulate study from the worker's
  `simulation_study` if the project has none and runs it through `run_study`, so
  replacing its results asks first like any study. **Studies are per folder but
  a spec is per file**: the study is `simulation` for `flowsheet.yaml` and
  `simulation_<stem>` for any other spec (`simulation_name`), an existing one
  whose listed `flowsheet` is another file is pointed back, and a frozen one is
  refused (it would run its snapshot, not the drawing). The study dialog sends
  a new study's `flowsheet:` as the open spec's path and keeps an edited
  study's own (the form has no such field, so `study_from` would drop it). **A measured channel may be a measure
  expression** (`"BOD5 = 0.65*[SS] + 0.65*[XS]"`, see `.claude/rules/plant-analysis.md`):
  the channels widget is a `VCombobox`, not a `VSelect`, so free text becomes a
  chip and travels as a plain string the draft must not touch — aquakin reads it
  when the study is written (`create_study` / `update_study`), and the refusal
  shows beside the buttons like any other. Do **not** add a per-keystroke
  validation round trip for it (the form rule above); `aquakin_ui` cannot read an
  expression itself anyway. A structured measure read from a file still rides in
  the row's `measures` map, keyed by name. A run started this way opens its
  results when it finishes (`_show_when_done`). **Results** (`show_results`) asks the worker for
  `study_results`, whose own `reasons` make the out-of-date note; the charts are redrawn in Python when the stream or channels
  change (`choose_results_stream` / `show_charts`).
- **The flowsheet's file** (`_spec_file`) is the one last opened or saved here;
  the toolbar path is only where Open reads and Save writes next (an unsaved
  new flowsheet has none, and Save, Simulate and New study ask for one rather
  than writing into the server's working folder). `_project` follows the file,
  never a half-typed path.
- **Unsaved work is never dropped silently** (`_unsaved`, against the `_saved`
  snapshot): Open asks (`request_open` → `_unless_unsaved` → `answer_unsaved`,
  "Save first" saving to the file, not to the newly typed path); Run and
  Simulate save the drawing before a live study runs the file.
- **Undo** (`_remember` before every document edit, `undo`, `UNDO_DEPTH`
  snapshots of `Flowsheet.snapshot`, cleared on Open). Delete removes; Backspace
  is not a delete key (a "go back" habit, and it reached the canvas behind a
  dialog). A widget change needs the bundle rebuilt.
- **Unapplied form edits:** every form widget sets `sel_dirty`; the canvas's
  select event carries `sel_rows` while it is set, and `on_select` applies them
  to the unit selected until then (or keeps it selected, with the field-named
  error, when they cannot be read). Save, Open, Simulate and Run carry them
  too (`_apply_pending`) and wait when they cannot be applied.
  `refresh_selection` clears the flag. When the rows Python computes equal the
  last ones sent, trame sends nothing and the page keeps its edits; even
  `state.dirty` did not reach the page, so a reset of the form (`cancel_apply`)
  marks the first row with a new `revision` (only there: a background redraw
  must never overwrite typing).
- **Aeration's species** is a form row (`aeration.species`, the model's
  species) while aeration is on; a stored species the model lacks (ASM1's `SO`
  on an ASM3 plant) is replaced by the model's oxygen (`forms._oxygen`), and
  `reshape_form` applies such suggestions at once.
- **A deleted unit and its wires are one removal**: Vue Flow reports them as
  two changes; the canvas queues them and emits one `remove` (one Undo).
- **A cancelled run is reported cancelled** even when it is killed before a
  checkpoint (`StudyRun.cancelled`). "Replace results?" is asked, and
  unfreezing asks, only for a study with results (`has_results`).
- **Selecting a unit opens its form** (`on_select` sets the Unit tab). The
  palette marks kinds with a required setting the form cannot edit
  (`_needs_python`: an SBR's phases, a dosing unit's reagent). A new flowsheet
  runs on days (`options: {time_unit: d}`). Errors reach the page without
  exception class names (the worker sends them apart, see the protocol above).
- **A scheduled value is text, read by aquakin** (`scheduled_number`,
  `scheduled_species_values` — an influent's `Q`, `T` and each `composition`
  entry): `forms._scheduled` keeps a number a number (so a constant feed is
  written to the spec exactly as before expressions existed) and anything else
  the text the user typed. `aquakin_ui` cannot read an expression — the
  independence rule — so an unreadable one is reported by the background check
  or by Save, never parsed here.
- **NaN is never a value:** `forms._finite` refuses `nan` and overflowing text;
  `inf` / `-inf` remain for "no limit" number settings.
- Icons are sent once (`icons` prop, injected into nodes); the spec text only
  while the Spec tab is open.
- **Removing a wired port asks first:** `Editor.apply_properties` previews with
  `Flowsheet.dropped_by` and, when connections would go, opens a dialog listing
  them; nothing changes until `confirm_apply` (`cancel_apply` restores the form).

## The widget build

`ui-widget/` holds the Vue source; `npm install && npm run build` writes
`aquakin_ui/widget/serve/` (`AK_DEV=1` keeps Vue Flow's development warnings,
which the production build strips; read them when edges vanish). Vue is not
bundled: trame-client provides it (3.5), hence the `trame-client` floor in the
`ui` extra. Commit the rebuilt bundle with any widget change: the build is
reproducible byte for byte from `package-lock.json`, and the `widget` CI job
(path-filtered, not required) fails when the committed bundle is not a fresh
`npm ci && npm run build`. It also builds the wheel and runs
`scripts/check_ui_package.py`, which fails unless the wheel carries the bundle
and every `icons/*.svg` byte for byte (the release build runs it too). A new
file the editor needs at run time goes in `REQUIRED` there.

## Testing

`tests/ui/`: independence, the document and forms (pure), the worker client
against a real worker or a stand-in process (`command=`), and an `Editor` built
and driven without a browser. An `Editor` test that needs to count requests or
make a command fail passes `worker=tests/ui/inprocess.InProcessWorker()`
(aquakin's own `serve`, answered in the test process, with `sent` and `refuse`);
one that needs the page's loop (background checks, `_off_page`, run events)
runs inside `asyncio.run`, and a `state.change` handler fires only after
`state.ready()`. Each sync rule Python owns has an `Editor` test
(`test_editor_sync.py`, `test_editor_runs.py`, `test_editor.py`).

`tests/browser/` serves the editor (`python -m aquakin_ui --no-browser --port 0`)
and drives it in Chromium with Playwright; it holds the canvas's rules (the diff,
handle validation, no echoes, one removal, Delete not Backspace, the form sent on
Apply). It is not in the required gate and is skipped without the `playwright`
package. Run it (`pytest tests/browser`) and add to it when a change touches
the canvas or the form; a widget change needs the bundle rebuilt first. Read
Python's document from the header chip ("10 units · 11 connections") and
messages through a recorder (the snackbar closes after four seconds), never by
a negative check with no sync point after it.
