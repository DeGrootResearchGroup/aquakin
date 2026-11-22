`Plant.check_units()` checks a plant's units before you solve it and
returns a `PlantUnitsReport`. On a plant created without `time_unit` it
confirms that the kinetic models agree on their time unit, and it reports a
solids species the thickener or Takács settler could not put on their
`g/m3` basis (one declared in moles, say).
Comparisons account for scale, so `mg/L` matches `g/m3` but `kg/m3` does
not. `plant.check()` includes the report.
