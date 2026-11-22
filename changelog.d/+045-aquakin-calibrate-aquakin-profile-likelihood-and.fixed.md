`aquakin.calibrate`, `aquakin.profile_likelihood` and `Plant.calibrate`
take `observation_units={"SNH": "mg/L", "XS": "kgCOD/m3"}` for data measured
in units other than the model's. The named columns, and an absolute `sigma`
in the same units, are converted to the model's units before the fit, so the
results are in the model's units. A unit of another quantity raises.
