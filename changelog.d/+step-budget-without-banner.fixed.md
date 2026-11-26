A reactor or plant solve that runs out of integrator steps raises aquakin's
`RuntimeError` naming the remedies without first printing a JAX callback
traceback and Equinox's banner to the log. This covers the batch, plug-flow,
particle-track, CFD and biofilm reactors, located events and the plant's forward
solve and steady-state fallback; the plant's `forward_fast` solve and the
forward-sensitivity solves still print it. Under `jax.grad` or `jax.jit` the
solve fails as before.
