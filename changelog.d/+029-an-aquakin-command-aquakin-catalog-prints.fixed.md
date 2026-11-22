An `aquakin` command. `aquakin catalog` prints the catalogue as JSON, and
`aquakin serve` answers JSON-line requests on standard input and output
(`catalog`, `models`, `model`, `check_unit`, `check_spec`, `ping`), so a separate process
such as a flowsheet editor can use aquakin without importing it.
