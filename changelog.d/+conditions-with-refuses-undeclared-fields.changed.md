`SpatialConditions.with_` (and so `model.default_conditions().with_(...)`)
refuses a field the conditions do not carry, with a `ValueError` naming it and
listing the declared fields. It used to add the field silently, so a misspelt
name (`with_(Temp=283.15)`) left the intended condition at its default with no
sign anything was wrong. Build conditions with a new field with
`SpatialConditions.uniform(...)`.
