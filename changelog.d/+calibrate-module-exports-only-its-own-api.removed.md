`aquakin.integrate.calibrate` no longer re-exports names it does not define.
Its `__all__` is now `calibrate`, `CalibrationResult`, `FreeICConfig`,
`LaplaceConfig`, `OptimizerConfig` and `PredictiveBand`. Import
`DifferentiationConfig`, `check_finite_gradient` and `forward_adjoint` from
`aquakin` (they were always public there). `native_time_factor`,
`to_unconstrained`, `from_unconstrained`, `dphysical_dunconstrained`, `observe`,
`observation_size` and `observation_takes_time` were internal helpers and are
no longer importable from this module; `Reactor` and `with_adjoint` are still
importable from it but are no longer listed as its API.
