# Flowsheet canvas prototype

A prototype of a drag-and-drop flowsheet editor built in trame (the stack
`aquaflux-ui` uses), with Vue Flow wrapped as one custom widget. Not part of the
`aquakin` package and not imported by it; the editor itself imports neither
aquakin nor JAX.

- `js/`: the widget source (Vue 3 + Vue Flow). `npm install && npm run build`
  writes the bundle into `trame_flowsheet/serve/`; `AK_DEV=1 npm run build` keeps
  Vue Flow's development warnings.
- `trame_flowsheet/`: the Python widget and the pre-built bundle, so running it
  needs no JavaScript toolchain.
- `aquakin_worker.py`: the client for one long-lived `python -m aquakin serve`
  process (JSON lines; requests serialized, replies read on a thread so a
  timeout works; a dead or stuck worker is restarted on the next request).
- `app.py`: the editor. Its palette, unit settings, help text, defaults and
  ports all come from aquakin's unit catalogue (`aquakin.plant.catalog`) through
  the worker. It starts on a BSM1-like layout.
- `icons/`: original line icons, one per unit kind (see `icons/README.md` and
  `icons/contact_sheet.png`).
- `screenshots/`: the editor driven by Playwright in headless Chromium.

Run it (with aquakin installed in the same environment):

```bash
pip install trame trame-vuetify
python app.py --port 8080
```

## What the editor does

- Units are dragged from a palette grouped by category; ports follow the
  catalogue's port rules as settings change (a mixer's inputs, a splitter's
  outputs).
- The Unit panel is generated from the kind's settings: a model picker,
  species pickers from the chosen model, the model's conditions pre-filled
  from its defaults, aeration as a mode plus its value, defaults shown as
  placeholders, required settings starred, and the docstring help of the focused
  field. Settings only Python can give yet are shown as such.
- Every unit is built by aquakin (`check_unit`) whenever its settings change.
  The Checks tab lists aquakin's own error messages and unfed inputs (errors),
  and the unconnected outputs, which are the plant's product streams, as
  `Plant.check()` reports them.

## Design points the prototype settled

- Python owns the flowsheet. The canvas draws what it is sent and reports only
  finished edits (`connect`, `move`, `remove`, `dropped`, `select`); dragging,
  panning and wire drawing stay in the browser.
- Python's updates are applied to Vue Flow's store as a diff (add / remove /
  update in place). Replacing the whole node or edge list rebuilds every node
  and re-validates every edge, which drops them.
- Connection checks while a wire is drawn sit on the port handles, not on the
  canvas-wide `is-valid-connection`, which Vue Flow also applies to loaded edges.
- Python marks its selected unit; removals and selection changes the canvas
  makes while applying Python's own update are not echoed back.
- The form is edited in the browser and sent whole on Apply (or when a choice
  changes the form's rows), so no keystroke waits on the server or is
  overwritten by it. Help sits in a fixed panel, not under each field: a hint
  that collapses on blur moves the Apply button between press and release.
- Recycle (back) edges are found in Python and drawn in their own lane below
  the units they span.
