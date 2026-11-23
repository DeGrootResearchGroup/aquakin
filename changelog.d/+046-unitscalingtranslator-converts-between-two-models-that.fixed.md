`UnitScalingTranslator` converts between two models that list the same
species but declare some of them in different units. `Plant.connect` and
`Plant.add_influent` insert it automatically for such a pair, and raise
`WiringError` when a pair of units cannot be converted.
