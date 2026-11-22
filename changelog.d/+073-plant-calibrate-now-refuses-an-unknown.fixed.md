`Plant.calibrate()` now refuses an unknown `OptimizerConfig(method=...)` with
the same `ValueError` as `aquakin.calibrate()`, naming the valid methods.
Before, the plant fit started and then failed inside SciPy with an unhelpful
`TypeError: 'NoneType' object is not callable`. The method is now checked when
the `OptimizerConfig` is built, so an invalid one is refused before any fit.
