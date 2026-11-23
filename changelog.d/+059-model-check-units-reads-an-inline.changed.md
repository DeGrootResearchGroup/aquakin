`model.check_units()` reads an inline half-saturation constant written as
`c / (c + X)` as a dimensionless inhibition term, so the WATS sewer models
no longer report false findings. `mg` and `mmol` are now parsed as units
rather than as a metre times a gram or mole.
