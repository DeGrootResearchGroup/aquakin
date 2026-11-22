`model.check_units()` also checks each rate against the species whose
stoichiometric coefficient is a fixed +1 or -1: such a species changes by
exactly the rate, so the rate must be in its concentration per time. This
catches a rate constant declared in the wrong units, such as a specific
phosphorus uptake rate written as 1/d where the stoichiometry is per g P.
Turn it off with `check_units(check_stoichiometry=False)`.
