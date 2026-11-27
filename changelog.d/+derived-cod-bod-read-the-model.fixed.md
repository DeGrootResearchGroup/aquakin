`plant.sludge_age` and `sludge_metrics` work on ASM2d and ASM3-family plants
such as `build_a2o`: the default `substrate="BOD"` raised `KeyError: 'SS'`, and
`substrate="COD"` counted only the COD species named as in ASM1 (`SI`, `XI`,
`XS` on ASM2d), so the F:M ratio was too low. `derived_COD` now sums every
species that carries COD according to the model's composition (or its declared
units), and `derived_BOD` takes 0.25 × the biodegradable COD, with each
species' inert share read from the model's stoichiometry; it accepts `params=`
so the share follows a calibrated `fXI`. A model whose stoichiometry does not
fix that share raises and points to `substrate="COD"`. ASM1, BSM1 and BSM2
results are unchanged.
