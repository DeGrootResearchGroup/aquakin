A plant created without `time_unit` whose kinetic models use different time
units, for example an ozonation model in seconds next to ASM1 in days, now
raises `UnitsError` when solved; give it a `time_unit` and the models are
converted to it. Such a plant was integrated against one clock and gave
wrong results without a warning.
