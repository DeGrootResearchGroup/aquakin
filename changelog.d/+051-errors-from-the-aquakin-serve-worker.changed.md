Errors from the `aquakin serve` worker, from `aquakin.plant.catalog.check_unit`
and from `aquakin.plant.spec.check_spec` now carry the message and the
exception's class apart, `{"error": "volume must be positive", "type":
"ValueError"}`, instead of `"ValueError: volume must be positive"`; the
`aquakin study run --json` error event does the same. The flowsheet editor
shows the message as it stands.
