`SensitivityResult.ranked_params()` now ranks parameters by their elasticity
`(dy/dp)·p/y`, the relative change in the output per relative change in the
parameter, rather than by the raw gradient, whose size depends on each
parameter's units. The new `SensitivityResult.elasticities()` returns the
values, and `ranked_params(elasticity=False)` gives the previous ranking.
`SensitivityResult` now also records the `params` it was evaluated at.
