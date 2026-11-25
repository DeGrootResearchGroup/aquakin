Results can be read back in the units you want: `C_named`, `C_named_many`,
`final_named`, `to_dataframe` and `to_csv` take `units=`, on reactor
solutions, plant solutions and plant streams. Pass one unit for every species
read (`sol.C_named("SNH", units="mg/L")`) or a `{species: unit}` dict for some
of them, the rest staying in the model's units; tables label their columns and
`df.attrs["units"]` with the units used. Only the scale is converted, so a unit
of another quantity (moles where the model has grams) raises. The conversion is
a multiplication, so a value read this way stays differentiable.
