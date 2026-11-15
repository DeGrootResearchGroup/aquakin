# Flowsheet canvas prototype

A throw-away prototype testing whether a drag-and-drop flowsheet editor can be
built in trame (the stack `aquaflux-ui` uses) by wrapping Vue Flow as one custom
widget. Not part of the `aquakin` package and not imported by it.

- `js/` — the widget source (Vue 3 + Vue Flow). `npm install && npm run build`
  writes the bundle into `trame_flowsheet/serve/`; `AK_DEV=1 npm run build` keeps
  Vue Flow's development warnings.
- `trame_flowsheet/` — the Python widget and the pre-built bundle, so running it
  needs no JavaScript toolchain.
- `app.py` — a demo editor on a BSM1-like layout: palette, canvas, unit
  properties, checks (open ports) and the flowsheet spec as it would be saved.
  It does not import `aquakin` or JAX; the unit catalogue is hand-written.
- `screenshots/` — the demo driven by Playwright in headless Chromium.

Run it:

```bash
pip install trame trame-vuetify
python app.py --port 8080
```

Design points the prototype settled:

- Python owns the flowsheet. The canvas draws what it is sent and reports only
  finished edits (`connect`, `move`, `remove`, `dropped`, `select`); dragging,
  panning and wire drawing stay in the browser.
- Python's updates are applied to Vue Flow's store as a diff (add / remove /
  update in place). Replacing the whole node or edge list rebuilds every node
  and re-validates every edge, which drops them.
- Connection checks during drawing sit on the port handles, not on the
  canvas-wide `is-valid-connection`, which Vue Flow also applies to loaded edges.
- Removals the canvas makes while applying Python's own update are not echoed
  back as user deletions.
- Recycle (back) edges are found in Python and drawn in their own lane below
  the units they span.
