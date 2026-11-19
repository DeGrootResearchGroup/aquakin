---
paths:
  - "aquakin_ui/**"
  - "tests/ui/**"
  - "ui-widget/**"
---

# Rules — `aquakin_ui/` (the browser flowsheet editor, `aquakin-ui`)

A local browser flowsheet editor: palette, Vue Flow canvas, a unit form generated
from aquakin's unit catalogue, aquakin's checks, and Open/Save of flowsheet specs
(`Plant.to_spec` / `Plant.from_spec`, see `.claude/rules/plant.md`). trame +
Vuetify, the same stack as aquaflux's `aquaflux-ui`. `ui` extra; entry point
`aquakin-ui [flowsheet.yaml] [--port] [--no-browser]` (`aquakin_ui.__main__:main`).

## ⚠️ The independence rule (binding, mechanically checked)

**`aquakin_ui` imports neither `aquakin` nor `jax`; `aquakin` never imports
`aquakin_ui`.** Everything about units, models and plants comes from one
long-lived `python -m aquakin serve` worker (`aquakin/__main__.py`, client
`aquakin_ui/worker.py`). Pinned by `tests/ui/test_independence.py`
(fresh-interpreter import of every module + AST scan of `aquakin/`). Tests may
import aquakin (to build expected values).

## Structure

| module | holds | notes |
|---|---|---|
| `worker.py` | `AquakinWorker`, `WorkerError` | one process, requests serialized by a lock, replies read on a daemon thread; a fresh worker's first reply gets `START_TIMEOUT` extra (importing JAX); reply ids checked; stderr to a per-worker log file (`mkstemp`, removed on `stop`) quoted in errors |
| `catalog.py` | `Catalog` | the worker's catalogue reply; `ports_for` reads the catalogue's port rules (no round trip) |
| `document.py` | `Flowsheet`, `Unit`, `edge_id` | **pure**: units keyed by stable id (the plant's unit name), labels as display text; `from_spec`/`to_spec` (format 2), `connect` (refuses with a reason), `dropped_by` (previews the connections new settings would remove) and `apply`, `remove` (also drops streams/endpoints naming the unit), `port_status`, `back_edges` (aquakin's recycles when known, else a DFS), `lanes`, `auto_layout` (wraps into bands of 7 ranks) |
| `forms.py` | `rows_for`, `read_rows` | **pure**: one row per catalogue setting (conditions and aeration expand); read-only settings and the aeration fields the form does not show are kept on read-back. Lists no setting or type of its own |
| `icons.py` | `ICONS`, `COLOURS`, `CATEGORY_LABELS`, `SVG`, `icon_name` | the only per-kind presentation table; an unlisted kind gets the tank icon |
| `study_form.py` | `fields_for`, `draft_for`, `study_from`, `missing` | **pure**: the study dialog's draft from the worker's `study_form` (numbers kept as text, an even time grid as start / stop / count) and back, empty fields and defaults left out; `missing` names a kind's required fields by label. Lists no setting of its own |
| `results.py` | `fmt`, `steady_table`, `channels_of`, `line_chart` | **pure**: the results view's tables (a column per stream, a row per channel with its unit) and SVG charts (one channel each, so never two y-axes; the dataviz palette's slot-1 blue, `#3987e5` in dark mode, set in `PAGE_CSS`; a CSS-hover crosshair and a native `<title>` tooltip per sampled time, at most `MAX_HOVER` of them, the line thinned to `MAX_POINTS`) |
| `jobs.py` | `StudyRun` | one study run = one `python -m aquakin study run --json` process: JSON progress events on stdout, `cancel` on stdin (killed after `CANCEL_GRACE` if no checkpoint comes); its own temporary log |
| `app.py` | `Editor` | the page; turns finished canvas edits into document changes; the Studies tab |
| `widget/` | `Flowsheet` (trame element), `module.py`, `serve/` | the built Vue Flow bundle, committed; source in `ui-widget/` |

Icons (`icons/*.svg`) are original line icons (see `icons/README.md`): 24-unit
viewBox, `currentColor` strokes, one shared vocabulary. Keep new ones to it.

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
  its results, then goes ahead (owner's decision).
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
  then, off the page's thread, creates the `simulation` study from the worker's
  `simulation_study` if the project has none (pointing at the spec when it is not
  `flowsheet.yaml`) and runs it through `run_study`, so replacing its results
  asks first like any study. A run started this way opens its results when it
  finishes (`_show_when_done`). **Results** (`show_results`) asks the worker for
  `study_results`; the charts are redrawn in Python when the stream or channels
  change (`choose_results_stream` / `show_charts`).
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
`ui` extra. Commit the rebuilt bundle with any widget change.

## Testing

`tests/ui/`: independence, the document and forms (pure), the worker client
against a real worker, and an `Editor` built and driven without a browser. Drive
the real page with Playwright when a change touches the canvas or the form; every
sync rule above was found that way, not by the unit tests.
