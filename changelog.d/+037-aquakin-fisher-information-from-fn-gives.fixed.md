`aquakin.fisher_information_from_fn` gives the Fisher information of any
forward model written as a JAX function `fn(theta) -> y`, not only an
aquakin reactor. The Jacobian comes from AD, `noise="lognormal"` observes
`log y`, and `transforms=` evaluates the information in unconstrained
coordinates. With `groups=` it also returns each candidate measurement's own
information. The result works with `identifiability` and `design_criterion`
unchanged.
