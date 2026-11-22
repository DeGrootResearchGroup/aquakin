Influents can declare the units their data are in with `units=` on
`InfluentSeries`, `model.influent(...)`, `InfluentSeries.constant` and
`read_influent_csv`, e.g. `{"t": "h", "Q": "m3/h", "T": "degC", "SS":
"mg/L"}`, and the data are converted to the units the plant works in: time to
the model's time unit, flow to m3 per that unit, temperature to Kelvin, and
each species to the model's declared unit. Only the scale is converted: a
unit that measures another quantity (moles where the model has grams, g COD
where it has g N) or cannot be read is rejected. The BSM influent loaders
declare their units.
