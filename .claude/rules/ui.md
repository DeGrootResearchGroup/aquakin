---
paths:
  - "aquakin_ui/**"
  - "tests/ui/**"
  - "tests/browser/**"
  - "ui-widget/**"
  - "installers/**"
---

# Rules — `aquakin_ui/` (the browser flowsheet editor, `aquakin-ui`)

A local trame + Vuetify editor (`ui` extra; `aquakin_ui.__main__:main`): palette,
Vue Flow canvas, a unit form generated from aquakin's unit catalogue, aquakin's
checks, and Open / Save of flowsheet specs (`.claude/rules/plant-spec.md`).

## ⚠️ The independence rule (binding, mechanically checked)

**`aquakin_ui` imports neither `aquakin` nor `jax`, anywhere in a file (a
function-level import counts); `aquakin` never imports `aquakin_ui`.** Everything
about units, models and plants comes from one long-lived `python -m aquakin
serve` worker (`aquakin/__main__.py`, client `aquakin_ui/worker.py`), so the page
starts without JAX. `tests/ui/test_independence.py` pins it, finding every module
itself. Tests may import aquakin to build expected values.

## Types (pyright, mechanically checked)

`aquakin_ui` must stay at 0 errors under basic-mode pyright. Fix a type where the
value comes from (narrow a `None` on that path, pass a known value down, correct
a wrong annotation). An `assert` narrows only an invariant a reader can check on
the spot; a `None` a page event or a missing file can produce is handled and
tested, not asserted away.

## Structure

| module | what an edit must not get wrong |
|---|---|
| `worker.py`, `catalog.py` | the only worker client; the catalogue as plain data (ports without a round trip) |
| `document.py` | **pure** (no page, no worker); units keyed by stable id (the plant's unit name), labels display only; `from_spec` refuses a kind the catalogue lacks (it would break every redraw) |
| `forms.py` | **pure**; lists no setting, type or choice: every row follows from the catalogue. **Numbers are shown exactly** (`_float_text`): Apply reads every field back, so a rounded display changes values nobody edited. The aeration value is read only in the mode the form was drawn with; a unit's conditions are kept when its model could not be described; read-only settings and unshown aeration fields survive read-back. An unreadable field raises `FieldError` with its row key. Each row's `suffix` is the unit its help names, flows and times on the plant's clock. A choice that redraws the form carries `reshape` (the page reads the flag, never a key). A `choice` setting is a select, with `reshape` when another setting's `when` names it; a setting whose `when` does not hold (an SBR's holding volume without a storage feed) is not drawn and reads back unset, so switching mode does not leave it behind. A reagent is three rows (`reagent.model`, `.label`, `.composition`, its species from the reagent's own model); all empty reads back as unset. A `records` setting (an SBR's phases) is one `records` row whose fields come from the catalogue's `fields` (a box for a true / false field, a default / true / false choice for an optional one); every box and choice is read back, an empty text field is left out, an entry left blank is skipped, and an error names the entry and field. A `tagged` setting (the settling model) is a `<name>.kind` choice that redraws the form and a row per argument of that kind; switching kind keeps the arguments both take. A `number_map` (constants, settling parameters) is one `named` row of `{name, value}` entries; a `kla_signal` aeration's value is a signal's name, required except just after switching to that mode |
| `icons.py` | the only per-kind presentation table; an unlisted kind gets the tank icon |
| `study_form.py` | **pure**; lists no setting of its own (all from the worker's `study_form`) |
| `results.py` | **pure**; one channel per chart, never two y-axes; colours are the dataviz palette (dark-mode blue set in `PAGE_CSS`); `analysis_view` raises on a kind `ANALYSIS_VIEWS` lacks |
| `settings.py` | the user's preferences file (`AQUAKIN_UI_CONFIG` overrides); `tests/ui/conftest.py` points it at a temporary file, so a test never touches the user's |
| `process.py`, `jobs.py` | starting / discarding child processes (below); `StudyRun`, one `aquakin study run --json` process per run |
| `app.py` | `Editor`: state, drawing, undo, canvas events, the unit form, and the binding of every function below; do not grow it |
| `checks.py`, `files.py`, `studies.py`, `disk.py`, `page.py`, `study_page.py`, `unit_form.py` | bound `Editor` concerns: checks; Open / Save; studies; following the project folder as other programs change it; the page; the Studies tab; the unit form's redrawing choices and resending its rows |
| `repeated.py` | the one implementation of lists of records edited on the page (`bound`, `redraw`, `add_button`, `move_button`, `remove_button`), shared by the study dialog and the unit form (composition, named numbers, phase table) |
| `widget/` | the built Vue Flow bundle, committed; source in `ui-widget/` |

**`Editor` is split by concern, bound like `Plant`'s modules.** Each module holds
plain functions whose first parameter is `self: Editor` (imported under
`TYPE_CHECKING`), and `app.py` binds each under its method name (`run_study =
studies.run_study`), so callers, page events and tests see one `Editor`. Put new
behaviour in its concern's module and bind it. Do not use mixins (pyright rejects
a mixin method typed `self: Editor`). A builder that never touches the editor is a
plain function called directly (`_table`), not bound. Annotate in
`Editor.__init__` every attribute a moved function assigns, or pyright infers it
from the `None` there alone. `_worker_study` stays in `app.py` (the one call naming
its command by variable; `tests/ui/test_protocol.py` pins where it is).

New icons (`icons/*.svg`) keep to `icons/README.md`: 24-unit viewBox,
`currentColor` strokes, one shared vocabulary.

## Child processes (`start_piped` / `Piped.close`)

- **Start every child process through `process.start_piped`**, never another
  `Popen`: piped text stdin / stdout, stdout on a reader thread it keeps, stderr
  to its own `mkstemp` log (two processes never share one), a new session.
- **Discard every child through `Piped.close`, on every path that lets go of it**
  (stop, replacement, timeout, exit, a finished run). It kills, reaps, and only
  then joins the reader and closes stdout, because closing a pipe another thread
  is reading can block or raise. Do not reap a child or close its pipes by hand;
  a `kill` that only makes a run end (`StudyRun.cancel`'s) is fine, as the reader
  then discards it. `tests/ui/test_processes.py` makes `ResourceWarning` an error
  so a leak fails.

## The worker protocol (pinned by `tests/ui/test_protocol.py`)

The two sides share no import, so what they must agree on is pinned by a test
that imports both. Keep it that way rather than copying knowledge across:

- **Every command name the editor sends** (`worker.request("...")`,
  `_worker_study("...")`, always a literal) is one of `_handlers()` in
  `aquakin/__main__.py`. Add a new command there first.
- **A refusal is `{"error": message, "type": class name}`**
  (`aquakin.plant.errors.error_reply`; also `check_unit` / `check_spec` results
  and a run's `error` event). `WorkerError` carries the message as its text and
  `.type`. Show the message as it stands; never parse a class name out of text,
  and never show exception class names on the page.
- **Study state and paths come from the worker** (`list_studies`: `path`,
  `state`, `has_results`, `flowsheet`; `study_results` and a run's `done` event:
  `has_results`, `reasons`). Decide "has results" from `has_results`, never from
  a state's name. Never rebuild `studies/<name>/study.yaml` or
  `../../flowsheet.yaml` here: send a new study's `flowsheet:` as the spec's
  absolute path and aquakin stores it relative. `STATE_COLOURS` (`study_page.py`)
  is this side's one table of state names, pinned to aquakin's `STATES`.
- **Every study kind has a view** (`results.SIMULATE` + `ANALYSIS_VIEWS` equal
  aquakin's `KINDS`).

## Sync rules (each was a bug found in the browser)

- **trame sends a value only when it differs from the one it last pushed**
  (`state.dirty` cannot force it; a page's first state does not count). A value
  that must reach the page again unchanged carries a new `revision`
  (`_resend_rows`, `_show_kept_selection`). Push state set outside a page handler
  where it is set (`with self.state:`, `schedule_checks`), or a later change back
  to the last pushed value is dropped.
- **Python owns the flowsheet.** The canvas emits only finished edits (`connect`,
  `move`, `remove`, `dropped`, `select`); dragging and wire drawing stay in it.
- **Apply Python's updates as a diff** (`syncNodes` / `syncEdges` in
  `ui-widget/src/FlowsheetCanvas.vue`): replacing Vue Flow's lists rebuilds every
  node (losing its measured size) and re-validates every edge, dropping some.
- **Wire validation sits on the handles** (`UnitNode.vue`), never on Vue Flow's
  canvas-wide validator (the `<VueFlow>` prop, `useVueFlow` options or store).
  The canvas-wide one runs on every edge `addEdges` adds and refuses a used
  handle, so it would drop a wire Python holds (a file with two wires into one
  input). Python is the authority on loaded wires; `tests/ui/test_widget_source.py`
  pins the placement.
- **No echoes:** removals and selection changes the canvas makes while applying
  Python's update are not reported back. When Python keeps a selection the canvas
  did not make (unreadable form edits), `_show_kept_selection` sends it back.
- **A deleted unit and its wires are one removal** (one `remove`, one Undo),
  though Vue Flow reports two changes. Delete removes; Backspace is not a delete
  key (it reached the canvas behind a dialog).
- **The form is edited in the browser and sent whole** on Apply (or when the
  aeration mode / model changes the rows); never a per-keystroke round trip (it
  lost clicks and overwrote text). Help sits in a fixed panel, not under each
  field (a collapsing hint moved Apply between press and release).
- **Unapplied form edits are never lost:** every form widget sets `sel_dirty`;
  while it is set the canvas's select event carries `sel_rows`, and `on_select`
  applies them to the unit selected until then (or keeps it selected, with the
  field-named error). Save, Open, Simulate and Run apply them first
  (`_apply_pending`) and wait when they cannot be applied. A reset of the form
  (`cancel_apply`, and a choice that cannot be applied) marks its first row
  with a new `revision` (`_resend_rows`); only there: a background redraw must
  never overwrite typing, nor clear `sel_dirty` (the redraw that ends a round
  of checks keeps it, `_run_checks`).
- **A choice that redraws the form blocks the form until Python's rows
  arrive.** The page marks the choice's row `waiting` and the form is `inert`
  while any row is, because the rows Python sends replace whatever was entered
  meanwhile. So `reshape_form` must send rows back on every path, the form as
  it was when the choice cannot be applied (`_resend_rows` drops the mark).
- **Removing a wired port asks first** (`Flowsheet.dropped_by`; nothing changes
  until `confirm_apply`).
- **aquakin's checks run in the background** (`schedule_checks`, after
  `CHECK_DELAY` of quiet), each unit and the plant cached by its spec, so the
  canvas never waits; "None" is claimed only after a check finished. Read the
  document on the loop thread (`_unit_check`, `_plant_request`) and run only the
  worker call (`_ask_unit`, `_ask_plant`) in the executor. A page handler must
  not call the synchronous `_check_unit` / `_check_plant`.
- **Open and Save go through aquakin** (`read_spec` / `write_spec`); never write
  the spec file from `aquakin_ui`. Both run off the page's thread (`_off_page`).
- **Save refuses to overwrite work** (`_save_refusal`): each window is its own
  document, so it refuses a file whose sha256 changed since this window opened or
  saved it, and any other existing file. A change on disk (`reload_if_changed`,
  polled by `watch_disk`) reloads unless unsaved; then it asks once, no "Save first".
- **The flowsheet's file** (`_spec_file`) is the one last opened or saved here;
  the toolbar path is only where Open reads and Save writes next. A new
  flowsheet has none, and Save, Simulate and New study ask for one rather than
  writing into the server's working folder. `_project` follows the file, never a
  half-typed path.
- **Unsaved work is never dropped silently** (`_unsaved`, against the `_saved`
  snapshot; a move counts): Open asks (`_unless_unsaved`; "Save first" saves to
  the file, not the newly typed path); Run and Simulate save before a live study
  runs the file.
- **Studies:** a run is a `StudyRun` process, never a worker request (the worker
  kills a request after `TIMEOUT`, and a run must not hold up checks or Save).
  Apply run events on the page's loop (`call_soon_threadsafe`). A cancelled run
  is reported cancelled even when killed before a checkpoint.
  **Replacing results and unfreezing ask first** (`_ask` / `answer_study`), only
  for a study with results, and offer "Copy first" (owner's decision). Unfreezing
  always asks; replacing may be turned off ("Don't ask again", the Studies tab's
  switch), and both go through `Editor.set_ask_replace`: nothing watches the
  state, so code calls the method, never sets the state.
- **Lists of records inside page state are nested, and the page does not redraw
  on a nested change**: every widget bound into one uses `repeated.bound` and
  every add / remove ends with `repeated.redraw` (the buttons do), or rows do not
  appear. Do not write a second set of row buttons. `sel_rows` is
  `client_only` (`page._build`): its redraws stay in the page, so a background
  redraw still compares with Python's last rows and leaves typing alone.
- **The study dialog's draft** is such state. The draft is sent whole on
  Create / Save and written by the worker (`create_study` / `update_study`);
  errors show beside the buttons. Switching kind keeps the fields both kinds
  have. A new study's `flowsheet:` is the open spec's path; an edited study keeps
  its own (`study_from` has no such field and would drop it). Each field heading
  carries the `<input id>-label` id Vuetify's `aria-labelledby` points at.
- **A measured channel may be a measure expression** (`.claude/rules/plant-analysis.md`):
  the channels widget is a `VCombobox`, so free text travels as a plain string
  the draft must not touch; aquakin reads it when the study is written.
- **Simulate** runs the spec's study (from `simulation_study`) through
  `run_study`, so replacing results asks like any study. **Studies are per folder
  but a spec is per file**: one study per spec (`simulation_name`); one whose
  `flowsheet` is another file is pointed back, and a frozen one is refused (it
  would run its snapshot, not the drawing).
- **A scheduled value is text, read by aquakin** (an influent's `Q`, `T` and
  `composition` entries, a flow setpoint, a `scheduled_port_map`'s pumps):
  `forms._scheduled` keeps a number a number (a constant is written exactly as
  before) and anything else as typed. Never parse an expression here; the
  background check or Save reports it. A mapping field is split into entries by
  `forms._entries` at commas outside brackets only (`clip(t, 0, 1)` is one value).
- **NaN is never a value:** `forms._finite` refuses `nan` and overflowing text;
  `inf` / `-inf` remain for "no limit" number settings.
- **A stored aeration species the model lacks** is replaced by the model's
  oxygen (`forms._oxygen`), and `reshape_form` applies such suggestions at once.
- Icons are sent once; the spec text only while the Spec tab is open.

## The widget build

`ui-widget/` holds the Vue source; `npm install && npm run build` writes
`aquakin_ui/widget/serve/` (`AK_DEV=1` keeps Vue Flow's development warnings;
read them when edges vanish). Vue is not bundled: trame-client provides it, hence
the `trame-client` floor in the `ui` extra. **Commit the rebuilt bundle with any
widget change**: the `widget` CI job fails when it is not a fresh `npm ci && npm
run build` (reproducible from `package-lock.json`). `scripts/check_ui_package.py`
fails unless the wheel carries the bundle and every `icons/*.svg`; a new file the
editor needs at run time goes in its `REQUIRED`.

## Testing

- An `Editor` test passes `worker=InProcessWorker()` (`tests/ui/inprocess.py`)
  unless the subprocess is its subject (those share `real_worker`). A run whose
  ending is not under test uses `in_process_runs` (`hold()` keeps it running);
  keep one real run per ending end to end.
- A test that needs the page's loop runs inside `asyncio.run`; a `state.change`
  handler fires only after `state.ready()`.
- An `Editor` test cannot see what trame drops (the first sync rule): pin that
  in `tests/browser`, which serves the editor and drives it in Chromium with
  Playwright (the `browser` extra and `python -m playwright install chromium`;
  skipped without Playwright; not in the fast gate). Run `pytest tests/browser`
  when a change touches the canvas or the form, after rebuilding the bundle, and
  add a page's files beyond BSM1 to the job's path filter.
- In `tests/browser`, the `page` fixture waits for the start-up checks
  (`checks_finished`, `tests/browser/conftest.py`), since every worker request
  queues behind a round of checks. A test that starts a round and needs it over
  waits for it (`checks_finished`, or `checks_done` for the Checks tab), never
  with a longer timeout; read Python's document from the header chip ("N units
  · M connections · K recycles") and messages through a recorder (the snackbar
  closes after four seconds); never a negative check with no sync point after it.

## The installers (`installers/`)

`install-aquakin.sh` (macOS, Linux) and `install-aquakin.bat` (Windows) install
the editor with uv for people who do not use Python. **The two must install the
same thing**: the same `PYTHON_VERSION`, the same uv options, and, once
`scripts/stamp_installers.py` fills in `AQUAKIN_VERSION` at release, that
release's `constraints.txt`, so a colleague gets the versions CI tested.
`tests/unit/test_installers.py` compares them and runs the shell one against a
stand-in uv; the batch file cannot run in CI, so test a change to it on Windows
and say so. The batch file is checked out with CRLF (`.gitattributes`); cmd
misreads one with bare LF.
