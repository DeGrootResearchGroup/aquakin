`Plant.to_spec()` and `Plant.from_spec(spec)`: a plant as a JSON/YAML file and
back. A spec (format 2) writes every setting, defaults included, with the
aquakin version that wrote it, so a later change of default cannot change a
saved plant. It lists each unit by its stable name with an optional display
`label` (`Plant.labels`), the connections as `"unit.port -> unit.port"` (or
with a state translator or recycle seed), constant influents, named streams,
endpoints and recycle options, and every model under `models:`, either shipped
or a model YAML file next to the spec, each with a content fingerprint; reading
a spec whose model has changed since warns (`SpecWarning`). Specs are validated
strictly: an unknown or misspelled setting, a value of the wrong type or a
model that is not listed is an error naming where it is, and YAML is read with
YAML 1.2 rules (`1e-5` is a number, `on` is text). A rebuilt BSM1 has a
bit-identical right-hand side. What a spec cannot express yet (a heat-balance
temperature model, array-valued settings) raises a `SpecError` naming it.
Controllers derived from a closed-loop `Aeration` or feedback `DosingUnit` are
not written; they are derived again. Writing a spec to a path
(`aquakin.plant.write_spec`, also exported with `load_spec_file`, `check_spec`,
`spec_inputs` and `spec_changes`) checks everything before touching the disk,
copies a model file from outside the project into `models/` together with
every file it `extends:`, and replaces the spec atomically. NaN is refused as a
setting everywhere (`inf` stays allowed where it means "no limit"), and a model
file may not take the name of a shipped model.
