Rate expressions gain four general math functions — `exp(x)`, `log(x)`,
`sqrt(x)` and `min(a, b)` — for rate laws the activated-sludge vocabulary
does not cover. `exp` and `log` matter most: a term such as a Beer–Lambert
attenuation `exp(-alpha * {z})`, a thermal-inactivation factor or a
logarithmic driving force previously had to be computed outside the model
and injected as a condition field, which put its coefficients outside the
parameter vector and so outside calibration, sensitivity and
identifiability. Written in the rate expression they are ordinary model
parameters that `jax.grad` reaches through a solve. `sqrt` is the clearer
spelling of half-order kinetics than `[X] ** 0.5`, and `min` is the ceiling
counterpart of the existing `max`. `log` and `sqrt` are guarded so their
gradient stays finite at a fully depleted species, where an unguarded
derivative is infinite and would make the whole parameter gradient
non-finite; `sqrt(0)` is `0`, while `log` keeps the mathematically honest
`-inf` there, so a model using it must keep its argument positive.
`check_units` requires the argument of `exp` and `log` to be dimensionless,
which catches an attenuation coefficient in the wrong units, and treats
`sqrt` as halving the argument's dimension. The shipped `uv_h2o2` model
deliberately keeps its fluence rate as a condition field — attenuation there
is a property of the reactor geometry, not the chemistry — and now says why.
