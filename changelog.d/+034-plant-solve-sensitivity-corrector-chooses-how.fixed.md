`Plant.solve_sensitivity(corrector=...)` chooses how each implicit stage of
the `[y; S]` sensitivity solve is solved. The new default, `"staggered"`,
converges the plant state first and then solves the sensitivity, which is
linear in `S`, with the same factorization. It forms no second derivatives,
so each sensitivity parameter adds only Jacobian-vector products to a step.
Its step controller limits step growth by the Newton contraction rate, so
fewer steps are wasted on Newton divergence. On a one-day BSM2 window with 16
parameters, a cached solve ran several times faster.
`"simultaneous"` is the previous diffrax augmented solve, which stays
differentiable. The two agree to the solver tolerance.
