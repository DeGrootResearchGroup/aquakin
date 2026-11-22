`aquakin.plant.plant_fisher_information` gives the Fisher information of a
plant design, the plant version of `fisher_information`. Measured channels
are named as for `Plant.calibrate`, including a unit's state such as
`("digester", ["S_IN", "S_gas_ch4"])`. `sigma` can be absolute or relative
to the signal, a `mask` drops unmeasured samples, and several runs add their
information. The sensitivity comes from `Plant.solve_sensitivity` (the
default, `mode="forward"`) or from the stable adjoint (`mode="reverse"`). At
a `Plant.calibrate` result with `loss="nll"` and the Gauss-Newton Laplace,
the reverse-mode matrix equals the Laplace Hessian.
