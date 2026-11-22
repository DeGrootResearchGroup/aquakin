The thickener (`IdealThickener`) and the Takács settler (with its default
BSM settling parameters) put the TSS of each solids species on the `g/m3`
basis of their built-in constants from the unit the model declares for it,
so a model with its particulates in `kgCOD/m3` thickens and settles exactly
as one in `g/m3`. A solids species declared in moles cannot be converted;
it is used as before and `Plant.check_units()` reports it.
