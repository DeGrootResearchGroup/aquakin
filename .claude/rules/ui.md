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
| `app.py` | `Editor` | the page; turns finished canvas edits into document changes |
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
  atomic replace), and the document is reloaded from what was written. Never
  write the spec file from `aquakin_ui` directly.
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
