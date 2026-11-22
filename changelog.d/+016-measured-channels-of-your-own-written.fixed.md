Measured channels of your own, written as a weighted sum: a channel may now be
given as the text `"BOD5 = 0.65*[SS] + 0.65*[XS]"` wherever a species or a
measurement keyword is accepted -- `plant.calibrate`, `plant_fisher_information`,
`plant_profile_likelihood`, `plant_rank_sensor_sets`, a study's `observables:`,
and the flowsheet editor's study dialog, where it is typed straight into the
channels field. Species go in brackets and the measurements (`COD`, `TKN`,
`NOx`, `TN`, `TP`, `TSS`, `Q`, `T`) are named as they are, so a channel can be
built on them as well as on species (`"sCOD = COD - [XS] - [XI]"`); their
weights keep following the composition parameters of the run, so a custom
channel built on `TN` moves with a fitted `i_XB` exactly as `TN` does. The
combination must stay linear -- a `*` needs a plain number on one side -- which
is what lets a channel remain a column of the measurement matrix. Results are
reported under the channel's name. `aquakin.plant.parse_measure` reads one in
Python, `aquakin.plant.Measure` gains a `keywords=` argument for the same thing
built by hand, and an unreadable expression raises
`aquakin.plant.MeasureSyntaxError` (a `ValueError`) when the study is written,
not when it runs.
