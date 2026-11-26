Unit strings accept the micro prefix on masses and amounts of substance:
`umol/L`, `µmol/L` (either the micro sign or the Greek mu) and `ug/L` are read
everywhere a unit is declared or converted (influent `units=`, observation and
output units, `check_units`, the speciation `molar_mass` audit). They were
unreadable, so an influent declared in `umol/L` was refused and a species in
`µmol/L` was skipped by the unit check.
