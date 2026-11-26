Plant gradients with the default `DifferentiationConfig(method="stable")` use
the sparse (colored) Jacobian in the backward pass from 50 states instead of
100, so plants of that size, such as BSM1 with its ideal clarifier, take it by
default; their repeated gradients, as in a calibration, are about 1.5 times as
fast and unchanged in value.
`plant.colored_jacobian_decision()` reports the choice, and
`IntegratorConfig(colored_jacobian=False)` keeps the dense backward.
