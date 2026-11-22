An influent that follows a pattern rather than a table:
`aquakin.plant.InfluentExpression` takes the flow, temperature and any species
as expressions of the plant's clock — `Q="18446 + 4000*sin(2*pi*t)"` — so a
diurnal or weekly load no longer needs a CSV generated beside the flowsheet.
Any value may equally be a plain number, and a species that is not named is
absent from the feed. The same values are accepted in a flowsheet spec's
`influent` unit (`Q`, `T` and each `composition` entry) and in the flowsheet
editor's unit form, where the whole feed stays text and no table file is
written. Expressions are read by `aquakin.plant.compile_time_expression`
(`aquakin.plant.time_expr`), compile to plain JAX arithmetic — so a solve
driven by one still differentiates — and offer `sin`, `cos`, `tan`, `exp`,
`log`, `log10`, `sqrt`, `abs`, `floor`, `min`, `max`, `mod`, `clip`, `step`,
`pi` and `e`. There is deliberately no comparison or conditional: a value that
switches uses `step` or `clip`, which stay differentiable. An unreadable
expression raises `aquakin.plant.TimeExpressionError` (a `ValueError`) when the
influent or spec is read, naming the field, rather than failing inside the
first solve.
