Plant observables can measure more than species: `"Q"` (flow), `"T"`
(temperature), `"COD"`, `"TKN"`, `"NOx"`, `"TN"` (`TKN + NOx`), `"TP"`,
`"TSS"`, or a `Measure` (any weighted sum of species, flow and temperature),
in `PlantObservable` / `SensorSet` channels. They follow what a laboratory
measures: dissolved oxygen, nitrate, hydrogen and methane are not COD,
dissolved N₂, N₂O and NO are in no nitrogen channel, and a digester's
headspace is in none. The composition weights follow the run's parameters, so
a fit of `i_XB` sees TN move with it. Calibration, plant Fisher information,
profile likelihood and sensor ranking all accept them. `StreamSeries` gains a
`T` trajectory for a stream that carries a temperature.
`composition_table` also serves a model read from a file that `extends:` a
shipped model with the same species, evaluated with its own parameters.
