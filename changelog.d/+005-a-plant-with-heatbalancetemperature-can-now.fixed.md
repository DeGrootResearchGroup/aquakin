A plant with `HeatBalanceTemperature` can now be solved inside `jax.jit` or
`jax.vmap`, so `aquakin.monte_carlo`, `aquakin.compare_scenarios` and
`aquakin.dgsm` work over it (they jit the function even with
`batched=False`). Building its initial temperatures converted a traced value
to a Python float and raised `ConcretizationTypeError`.
