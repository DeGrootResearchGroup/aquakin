`Plant.solve_sensitivity` now caches its compiled solve on the plant. Repeated
calls with new parameter values, initial state, time span, save times (of the
same length) or tolerances reuse it instead of recompiling, which helps
`dynamic_dgsm` samples, chained windows and parameter sweeps.
