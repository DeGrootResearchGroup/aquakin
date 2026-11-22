Plant states by name: every unit names its state entries
(`unit.state_names()`), and `Plant.state_names()`, `Plant.named_state(y)` and
`Plant.state_from_named(named)` turn a flat state into
`{unit: {name: value}}` and back, so a stored state survives a change of unit
order. **Custom units** implementing the `Unit` protocol should add
`state_names()`.
