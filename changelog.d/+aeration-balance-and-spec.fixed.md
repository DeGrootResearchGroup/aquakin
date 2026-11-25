`plant.mass_balance()` now counts the oxygen an aerated tank receives with the
aeration temperature correction on (`Aeration(temperature_correction=True)`) as
the solve does; it used the uncorrected saturation and kLa, so COD did not close
away from the reference temperature. Saving a plant (`Plant.to_spec`) no longer
drops an aeration's hand-listed stripped gases (`Aeration(stripped={...})`).
