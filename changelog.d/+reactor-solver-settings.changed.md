**Breaking:** every reactor (`BatchReactor`, `PlugFlowReactor`,
`BiofilmReactor`, `ParticleTrackReactor`, `CFDReactor`) now takes its solver
configuration as one keyword, `settings=aquakin.SolverSettings(...)`, as
`DispersedFlowReactor` and `TanksInSeries` do, and so does
`integrate_ensemble`. The loose `rtol=`, `atol=`,
`integrator=` and `diff=` keywords are removed, and passing one raises a
`TypeError`. Move them into `SolverSettings`, which has the same fields and
the same defaults, so results do not change: write
`aquakin.BatchReactor(model, conditions, settings=aquakin.SolverSettings(rtol=1e-8, atol=atol, integrator=aquakin.IntegratorConfig(dtmax=0.01)))`
where you wrote
`aquakin.BatchReactor(model, conditions, rtol=1e-8, atol=atol, integrator=aquakin.IntegratorConfig(dtmax=0.01))`.
`Plant.solve` keeps its `rtol=`, `atol=`,
`integrator=` and `diff=` keywords, and `calibrate` and `sensitivity` their
`diff=`.
