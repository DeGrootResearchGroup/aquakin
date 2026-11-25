Unit strings now read `dm` as the decimetre, so `mol/dm3` and `g/dm3` are
concentrations wherever a unit is converted (influent `units=`, calibration
`observation_units=`, translator and solids conversions): `mol/dm3` is exactly
`mol/L`. They were read as a day times a metre, which refused the conversion
or, with a time unit present, gave a wrong factor. A bare `d` is still the day
(`1/d`, `g/m3/d`), and an exponent may now also be written `**` (`dm**3`).
