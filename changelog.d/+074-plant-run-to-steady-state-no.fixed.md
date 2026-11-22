`Plant.run_to_steady_state()` no longer reports convergence in silence when its
default `criterion="norm"` has stopped early. That test compares `||dy/dt||`
with `ss_atol + ss_rtol*||y||`, so every state inflates the tolerance that
every other state is held to -- and an IFAS/MBBR tank defeats it, because its
held-fixed attached biomass contributes `n_layers * density` to `||y||` and
nothing to `||dy/dt||`. A solve could stop with the tank's bulk still moving by
70% of its own value per day and call it converged. The componentwise scaled
residual the result already carried is now checked against the verdict, and a
`RuntimeWarning` names the offending state and points at
`criterion="scaled_residual"`. No steady state changes: this is a check on the
verdict, not on the solve.
