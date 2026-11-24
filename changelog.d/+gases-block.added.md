A model can declare which of its species are dissolved gases in a new
`gases:` block, with Henry's-law solubilities and transfer coefficients from a
built-in gas library (O2, N2, CO2, CH4, H2) that each entry can override. The
declaration is available as `model.gases`; it does not change how any existing
model runs.
