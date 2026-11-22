`aquakin-ui` (`pip install "aquakin[ui]"`): a browser flowsheet editor. Drag
units from a palette, wire their ports, edit settings in a form generated from
each unit's own documentation, see aquakin's errors for each unit and for the
whole plant, and open or save flowsheet specs (saved by the same writer as
`Plant.to_spec`, so a model or influent file from elsewhere is copied into
the project). A Studies tab lists the project's studies with their status and
runs them in the background with progress and cancel. **Simulate** runs the
plant: it saves the flowsheet, sets up a `simulation` study the first time
(one per flowsheet file in the folder) and
shows its results, a steady state as tables of the streams leaving the plant
and a run in time as a chart per channel (**Results** on any study shows them
again; temperature is shown in °C). Fisher information, sensor-set ranking,
calibration (the data against the fit) and profile studies show their results
as tables and charts too. Unit forms show each setting's unit where its help
names one, plain help text, and an unreadable value under its field; new
settling units take the particulates of their model, and selecting a unit
opens its form. Replacing a study's results asks first,
with a **Don't ask again** box; the Studies tab's **Ask before replacing
results** switch turns the question back on, and the choice is kept for the
user between sessions. **New study** and
**Edit** set a study up in a dialog that offers the plant's own streams,
parameters and channels; Save refuses to
overwrite a flowsheet that changed on disk since it was opened; opening
another file with unsaved edits asks first, **Undo** takes back the last
edits, and running a study saves the drawing first.
