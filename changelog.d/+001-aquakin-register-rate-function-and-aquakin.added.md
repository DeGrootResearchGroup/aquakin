`aquakin.register_rate_function` and `aquakin.RateFunction`: add a rate
function aquakin does not ship, without editing the package — the rate-law counterpart of the `Unit` protocol
that already lets you add a plant unit operation. A rate law needing a
function outside the built-in set previously meant forking aquakin or folding
the term into a precomputed condition field, which put its coefficients
outside the parameter vector and so outside calibration, sensitivity and
identifiability. A registered function is not second class: it parses,
evaluates, reaches the same batched rate kernel — bit-identically to its own
scalar path — and differentiates exactly as a built-in does, with its
constants as ordinary model parameters. Register before the model that uses
it is loaded; a model file naming an unregistered function now fails with an
error listing what is available and naming the remedy. A name or kernel key
that collides with a built-in is refused, since shadowing `monod` would
change what existing model files mean, and only registration can add a
function — defining a node class is never enough on its own. A function may
optionally declare how its result's dimension follows from its arguments', so
`check_units` can check expressions using it. `registered_rate_functions`
lists what is registered. A rate function subclasses the public
`aquakin.RateFunction`, so extending aquakin as documented needs no private
name. Its `op` gets none of the built-in guards, so the guarded helpers it
can use — `safe_pow`, `safe_sqrt` and `safe_log` in `aquakin.core.kernels` —
are public too: a fractional power written as plain `x**n` has an infinite
derivative at a depleted species.
